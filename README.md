### ml-spark-sklearn-tensor collects my practice of projects written in Scikit-Learn (plus Panda, matplotlib and Numpy) drawn from the book "Hands-On Machine Learning with Scikit-Learn & TensorFlow" and port them to Apache Spark as much as I can  
#### The topics include:

    1. Chapter 2: End-to-End Machine Learning Project
       An excellent project to guide users through end-to-end processes of a Machine Learning Project: 
       'Discover and Visualize the Data to Gain Insights', 
       'Prepare the Data for MachineLearning Algorithm', 
       'Select and Train a Model' and 
       'Fine-Tune You Model'.  It instills the methodology of Machine Learning through hand-on practice rather than 
       pure theory (The book will introduces more theories in the following chapters)
       
       a) The reason industry ports machine learning projects to Apache Spark is to utilize its distributed computing 
          power when Spark is deployed in Hadoop Yarn cluster.   Google Tensor-flow which was open-sourced in late 2015
          was also a distributed platform. Tensor-flow is specialized in Artificial Neural Network and Deep Learning.  
          However, it is allowed plugged in with Linear Regression Model etc.
            
       b) Panda is built-in with plot, which can plot histograms and scatter etc.  There is no comparison in terms of 
          visualization when Panda combined with matplotlib.pyplot capability. They are irreplaceable as a tool to 
          gain first insight of data.  Spark cannot compete on this front.
                 
       c) Spark borrows at least SQLFrame terms from Panda.  It's very easy for Panda to load Housing data  No gimmick
          and no data manipulation.  I need to define schema and code the following to get Spark to work Housing data 
          as expected.
          
              val housing = spark.read.
                            option("header","true").schema(customSchema).csv("../datasets/housing").
                            filter(!isnull($"ocean_proximity"))
                            
          For some reasons, Spark does not filter out 'isnull' data of ' ocean_proximity'.  Without filtering,
          I will get counts execeeding 20640 which I got from Panada steadily.  Also the number I got from 
          spark kept flutuating.
          Without my customized schema, all fields are of StringType.  Adding option of "header") to tell Spark 
          the first line is heade not data.
          
       d) Panda use info, describe and head for basic statistics and Spark use printSchema, summary.show and show for 
          that. Panda use 'value_counts' to display group by counts in descending order.  Spark describe.show does not 
          include percentile: (25%, 50%(median) etc. 
          
          housing["ocean_proximity"].value_counts() //panda way
          housing.groupBy($"ocean_proximity").count().orderBy(desc("count")).show() // Spark way
          
       e) There is big difference how Panda and Spark view DataFrame
          In Panda, three important components of a DataFrame: 
          data: numpy ndarray
          columns: array-like
          indexes: array-like
          
          can be completely separate and manipulated individually.  That's why you can use any Numpy operations and 
          functions to manipulate data like the followings
          
              room_per_household = X[:, rooms_ix] / X[:, household_ix]
              population_per_household = X[:, population_ix] / X[:, household_ix]
              return np.c_[ X, room_per_household, population_per_household ]
          
          to add additional fields to DataFrame. You do need to be good at Numpy. rooms_ix etc. are indexes and np.c_
          combine data horizontally as long as data have the same numbers of row.  
          
          Spark does allow data and schema/ column name separately like
              spark.read.schema(customSchema).csv(......)
                             OR
              val housing_temp2 = housing_temp.rdd.map(....).toDF("label", "features", "op", "income_cat")   
          
          However, it follows SQL language to select, add additional fields like
              df.withColumn("room_per_household", $"total_rooms" / $"households").               
                                     
          Spark even has Spark SQL that make people from SQL background transit to Spark easily than Panda.  Adding 
          additional fields are super easy in Spark.  
           
       f)  There is no out-of-box StratifiedSplit implementation in Spark.  Scikit-Learn has StratifiedShuffleSplit in 
           model_selection package. However, you can put together StratifiedSplit strategy by using 
           Dataset.stat.sampleBy with fraction map.
           
                val fractions = housing_income_cat.select($"income_cat").distinct().rdd.map {
                    case Row(key: Double) =>
                      key -> 0.8
                }.collectAsMap.toMap
                
                val strat_train_temp = housing_income_cat.stat.sampleBy("income_cat", fractions, 42L)    
          
       g)  Spark borrows its Pipeline concept from Scikit-Learn.  In Scikit-Learn, there are
           Estimators:   fit (training DataFrame) to get insight/ statistics of data
           Transformers: transform (validate/ test DataFrame).  An estimator can be a a transformer too.  That's why 
                         Imputer has fit_and_transform method          
           Predictors:   predict (test data).  Again a estimator can be a predictor too.
         
           In Spark world, estimator and transformer are completely separate.  Transformer is also a predictor. In Spark 
           terms, model is a transformer.  Typically, an estimator and model are paired together.  For example, 
           
                class ALS extends Estimator[ALSModel].....
                class ALSModel extends Model[ALSModel]....
                
          Model is returned from Estimator.fit method.  Model.transform will add 'prediction' field to the DataFrame.  
          Therefore, model is also a predictor. Evaluation can apply to the DataFrame after model.transform.  The
          typical steps will be 
                
                 val model = als.fit(trainPlusDS, paramMap)
                 val prediction = model.transform(valDS)
                 val rmse = evaluator.evaluate(prediction) 
                       
       h) Preprocessings in Scikit-Learn can only apply to either numeric data or categorical data.  You have to
          separate data manually or dynamically select attributes.  You apply preprocessor to the whole DataFrame 
          and you cannot select specific fields to apply to.  Then you union pipeline together like the followings:
                          
                 num_pipeline = Pipeline([
                     ('selector', DataFrameSelector(num_attribs)),
                     ('imputer', Imputer(strategy="median")),
                     ('attribs_adder', CombinedAttributesAdder()),
                     ('std_scaler', StandardScaler())
                 ])
                 
                 full_pipeline = FeatureUnion([
                     ('num_pipeline', num_pipeline),
                     ('cat_pipeline', cat_pipeline)
                 ])                 
             
          Notice that DataFrameSelector is a class you have to program yourself and provide 'fit' and 'transform'
          method so that it can be an Estimator as well as a Transformer.
             
          Spark allows you to specify input columns and output columns to apply your pre-processors.  Then you apply 
          a Pipeline to the whole DataFrame.
             
                 val imputer = new Imputer().setStrategy("median").
                     setInputCols(Array("total_bedrooms")).setOutputCols(Array("total_bedrooms_out"))                     
                 val pipeline = new Pipeline().setStages(Array(imputer, indexer, encoder))    
             
          Housing has hybrid of numeric and categorical data.  It's handy to apply both numeric as well as 
          categorical stages to the Pipeline.   However, it's annoying to specify InputCols if you have a lot.  
                         
       i) Notice that I only apply Pipeline up to OneHotEncoder.  That's because I have to deal with Spark legacy 
          issue.  Spark RDD use org.apache.spark.mllib.regression.LabeledPoint and point is a 
          org.apache.spark.mllib.linalg.Vector.  MostSpark Regressors and Classifiers still only take Vector 
          (in ml instead of mllib) features.  StandardScaler estimator only take Vector data too.  Therefore, 
          I have to convert data to Vector:
             
                 val housing_temp2 = housing_temp.rdd.map(r => (r.getFloat(8), Vectors.dense(
                    r.getFloat(0), r.getFloat(1), r.getFloat(2), r.getFloat(3), r.getFloat(11), r.getFloat(5), 
                    r.getFloat(6), r.getFloat(7), r.getDouble(14), r.getDouble(15), r.getDouble(16)), 
                    r.getAs[Vector](13), r.getDouble(10))).toDF("label", "features", "op", "income_cat")
                 val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").
                    setWithStd(true).setWithMean(true)
                   :
                 //commbine numeric vector and catogorical-converted vector into one before I can apply Regressor
                 val combinedData = scaledData.rdd.map(r => (r.getFloat(0), new DenseVector( 
                    r.getAs[Vector](4).toArray ++ r.getAs[Vector](2).toArray), r.getDouble(3))).
                    toDF("label", "features", "income_cat")
                  
          I have to automate the process if I have lots of features.  This is probably the most difficult part 
          when you port Scitkit-Learn project to Spark
             
       j) Scikit-Learn allows users having more granule control over hyperparameters and options(for ex refit) for
          Machine Learning models etc.. Estimator, Transformer and Predictor also provide more insight info. 
          Spark CrossValidator is more or less like GridSearchCv in Scitkit-Learn.   However, GridSearchCv definitely
          is much informative.  One thing bothers me a lot is that Spark CrossValidator does not provide info of 
          the best params used by the best model of underline estimator.  In Scikit-Learn, I can get
             
                 grid_search.best_params_
                 {'max_features': 8, 'n_estimators': 30}  //return                                                          

          There is no equivalent method in Spark CrossValidator.  
             
                 cvModel.bestModel.asInstanceOf[ALSModel].extractParamMap()
                 
          The above only provide basic paramMap when you new ALS and does not provide extra parmaMap you use in 
          ParamGrid.
             
          The best I can get is
               
                 (lrModel.getEstimatorParamMaps zip lrModel.avgMetrics).minBy(_._2)._1
                 
          The first part returns Array       
                 Array(({
                 	linReg_92b756da9d32-elasticNetParam: 0.001,
                 	linReg_92b756da9d32-regParam: 1.0E-9
                 },69597.44343297838), ({
                 	linReg_92b756da9d32-elasticNetParam: 1.0E-4,
                 	linReg_92b756da9d32-regParam: 1.0E-9
                 	:
                 	:
                 
       k) This is the first time I use DecisionTree and RandomForest ensemble.  I find that it is very easy to overfit 
          the model.  I take suggestion from googling to pre-pruning that stop growing the tree.  On RandomForest,
          limit number of tree seems to increase overfitting.  I also have to limit maxDepth and 
          increase minInstancesPerNode from overgrowing the tree.  I definitely will improve once I know 
          DecisionTree and RandomForest theory a little better.  I have pretty good grasp of LinearRegression and
          LogisticRegression and PCA thanks to  "Distributed Machine Learning with Apache Spark (SQL, DataFrame)"
          course offered by edX.

#### How to use Jupyter with Spark:

    To get Jupyter to work in your workspace, please refer to Chapter 2: End-to-End Machine Learning Project ->
    Get the Data -> Create the Workspace
    
    To get Jupyter to work with Spark, you need to install spark. Download https://spark.apache.org/downloads.html, 
    untar the tar.gz file to the place desired.  You might need to set following environment vars for example. 
                 export SPARK_HOME=~/Public/spark-2.2.1-bin-hadoop2.7
                 export PATH=$PATH:$SPARK_HOME/bin
                 export PYSPARK_PYTHON_DRIVER=$ML_PATH/env/bin/python
                 export PYSPARK_PYTHON=$ML_PATH/env/bin/python
    
    The last two lines are to avoid python version incompatibility between the python that launches jupyter and the one 
    launches spark-submit.
    
    The best way to get Jupyter to work with Spark (edit spark or py-spark application and run in jupyter notebook) 
    is to install toree. However, the toree you installed from pip is most likely toree 0.1.0 which is built with 
    Scal 2.10.  Spark 2.x.x is built with Scal 2.11.  You will get unknown method Exception if you install that version
    of toree.  
    
    Instead, Download toree of 0.2.0 manually from the link https://pypi.anaconda.org/hyoon/simple/toree 
    then pip install it in your ML env space.  
                 pip install ~/Downloads/toree-0.2.0.dev1.tar.gz                 
                  
    Install toree with what intepretaters you need.  The following is an example    
                 jupyter toree install --user --spark_home=$SPARK_HOME --kernel_name=apache_toree -\ 
                 -interpreters=PySpark,Scala,SQL   
                 
    Launch jupyter notebook again, click new button, you should see pyspark, Scala and SQL options.  Of course, 
    you have to import necessary Spark or scala/ python libaraies needed. You can try  my Pi.ipynb under 
    spark or pyspark to start with.
                         