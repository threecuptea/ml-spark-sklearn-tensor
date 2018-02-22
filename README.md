### ml-spark-sklearn-tensor collects my practice of those hand-on projects written in Scikit-Learn (plus Panda, matplotlib and Numpy) from the book "Hands-On Machine Learning with Scikit-Learn & TensorFlow" and port them to Apache Spark as much as I can  
#### The topics include:

    1. End-to-End Machine Learning Project: An excellent project to guide users through end-to-end processes of a 
       Machine Learning Project: 'Discover and Visualize the Data to Gain Insights', 'Prepare the Data for Machine
       Learning Algorithm', 'Select and Train a Model' and 'Fine-Tune You Model'.  It instills the methodology of 
       Machine Learning through hand-on practice rather than pure theory (It introduces more theories in the following 
       chapters
       
       a) The reason industry ports machine learning projects to Apache Spark is to utilize its distributed computing 
          power when Spark is deployed in Hadoop Yarn cluster.   Google Tensor-flow which was open-sourced in late 2015
          was also a distributed platform. Tensor-flow is specialized in Artificial Neural Network and Deep Learning.  
          However, it is allowed plugged in with Linear Regression Model etc.
            
       b) Panda is built-in with plot, which can plot histograms and scatter etc.  There is no comparison in terms of 
          visualization when Panda combined with matplotlib.pyplot capability. They are irreplaceable as a tool to 
          gain first insight of data.  Spark cannot compete on this front.
                 
       c) Spark borrows at least SQLFrame terms from Panda.  It's very easy for Panda to load Housing data  No gimmick
          and no manipulation.  I need to define schema and code the following to get Spark to work Housing data 
          as expected (I was very diappointed) .
          
              val housing = spark.read.
                            option("header","true").schema(customSchema).csv("../datasets/housing").
                            filter(!isnull($"ocean_proximity"))
                            
          For some reasons, Spark does not filter out 'isnull' data of ' ocean_proximity'.  Without filtering,
          I will get counts of fields not NaN not only execeed 20640 which I got from Panada steadily but kept 
          flutuate.
          Without schema, all fields are of StringType.  Adding option of "header") to tell Spark the first line is 
          heade not data.
          
       d) Panda use info, describe and head for basic statistics and Spark use printSchema, describe.show and show for 
          that. Panda use 'value_counts' to display group by counts in descending order.  
          
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
          
          which you add additional fields to DataFrame. You do need to be good at Numpy 
          
          Spark does allow data and schema/ column name separately like
              spark.read.schema(customSchema).csv(......)
                             OR
              val housing_temp2 = housing_temp.rdd.map(....).toDF("label", "features", "op", "income_cat")   
          
          However, it follows SQL language to select, add additional fields like
              df.withColumn("room_per_household", $"total_rooms" / $"households").               
                                     
          Spark even has Spark SQL that make people from SQL background to transit to Spark than Panda.  Adding 
          additional fields are super easy in Spark.  
           
       f)  There is no out-of-box StratifiedSplit implementation in Spark.  Scikit-Learn has StratifiedShuffleSplit.
           However, you can put together StratifiedSplit strategy by using Dataset.stat.sampleData with fraction map.
           
                val fractions = housing_income_cat.select($"income_cat").distinct().rdd.map {
                    case Row(key: Double) =>
                      key -> 0.8
                }.collectAsMap.toMap
                
                val strat_train_temp = housing_income_cat.stat.sampleBy("income_cat", fractions, 42L)    
          
       g)  Spark borrows its Pipeline concept from Scikit-Learn.  In Scikit-Learn, there are
           Estimators:   fit (training DataFrame) to get insight/ statistics of data
           Transformers: transform (validate/ test DataFrame).  An estimator can be a a transformer too.  That's why 
                       Imputer has fir_and_transform method          
           Predictors:  predict (test data).  Again a estimator can be a predictor too.
         
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
             and you cannot select specific fields to apply to.  Then you union pipeline together like the 
             followings:
                 
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
             method so that it can be an estimator as well as a transformer.
             
             Spark allows you to specify input columns and output columns to apply your pre-processors.  Then you apply 
             a Pipeline to the whole DataFrame.
             
                 val imputer = new Imputer().setStrategy("median").
                     setInputCols(Array("total_bedrooms")).setOutputCols(Array("total_bedrooms_out"))                     
                 val pipeline = new Pipeline().setStages(Array(imputer, indexer, encoder))    
             
             Housing has hybrid of numeric and categorical data.  It's handy to apply both numeric as well as 
             categorical stages to the Pipeline.   However, it's annoying to specify InputCols if you have a lot.  
                         
       i) Notice that I only apply Pipeline up to OneHotEncoder.  That's because I have to deal with Spark legacy 
             issue.  Most Spark Regressors and Classifiers only take Vector features.  StandardScaler estimator only
             take Vector data too.  Therefore, I have to convert data to Vector:
             
                 val housing_temp2 = housing_temp.rdd.map(r => (r.getFloat(8), Vectors.dense(
                 r.getFloat(0), r.getFloat(1), r.getFloat(2), r.getFloat(3), r.getFloat(11), r.getFloat(5), r.getFloat(6), 
                 r.getFloat(7), r.getDouble(14), r.getDouble(15), r.getDouble(16)), r.getAs[Vector](13), r.getDouble(10))).
                 toDF("label", "features", "op", "income_cat")
                 val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").
                 setWithStd(true).setWithMean(true)
                   :
                 //commbine numeric vector and catogorical-converted vector into one before I can apply Regressor
                 val combinedData = scaledData.rdd.map(r => (r.getFloat(0), new DenseVector( 
                    r.getAs[Vector](4).toArray ++ r.getAs[Vector](2).toArray), r.getDouble(3))).
                    toDF("label", "features", "income_cat")
                  
             You have to automate the process if you have lots of features.  This is probably the most difficult part 
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
               
                 (lrModel.getEstimatorParamMaps zip lrModel.avgMetrics).sortBy(_._2).first._1
                 
       k) This is the first time I use DecisionTree and RandomForest ensemble.  I find that it is very easy to overfit 
             the model.  I take suggestion from what I google to pre-pruning that stop growing the tree.  On 
             RandomForest, limit number of tree seems to increase overfitting.  I also have to limit maxDepth and 
             increase minInstancesPerNode from overgrowing the tree.  I definitely will improve once I know 
             DecisionTree and RandomForest theory a little better.  I have pretty good grasp of LinearRegression and
             LogisticRegression and PCA thanks to  "Distributed Machine Learning with Apache Spark (SQL, DataFrame)"
             course offered by edX
