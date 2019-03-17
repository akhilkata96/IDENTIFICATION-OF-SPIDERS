package com.bdm.spark_scala
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import java.lang.String
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.elasticsearch.spark._
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint

object main
{
  def main(args: Array[String])
  {
    val conf = new SparkConf().setAppName(s"Book example: Scala").setMaster("local[2]").setAppName("project")
  //spark conf allows you to set some of common properties
    val sc = new SparkContext(conf)
       //spark context tells how application would be run(based on default or sparkconf set master)
    val input = sc.textFile("project.txt")
    //reading text file from default path
    val botspit = input.filter({bot => bot.contains("bot")})
    //creating an rdd to store urls aceessed by bots
    val notbot = input.filter({bot => !bot.contains("bot")})
    //creating an rdd to store urls that are accessed by humans
    botspit.foreach(println)
    //prints the data in rdd botspit
        val bots = botspit.map(line => line.split("\\s",2)(1).trim())
        //removing unwated data from each line by splitting with spaces and trimming
        val notbots = notbot.map(line => line.split("\\s",2)(1).trim())
        notbots.foreach(println)
        //removing unwated data from each line by splitting with spaces and trimming
   val tf = new HashingTF(numFeatures = 100)
                // Create a HashingTF instance to map url text to vectors of 100 features.
           // Each url is split into words, and each word is mapped to one feature.
    val spamFeatures = bots.map(w => tf.transform(w.split("")))
    //coverts bot url to vectors of 100 features
    val hamFeatures = notbots.map(w => tf.transform(w.split("")))
        //converts notbot rdd  to vectors of 100 features
     // Create LabeledPoint datasets for positive (bots) and negative (human) examples.
    val positiveExamples = spamFeatures.map(features => LabeledPoint(1, features))
    val negativeExamples = hamFeatures.map(features => LabeledPoint(0, features))
         spamFeatures.take(2).foreach(println)
    val trainingdata = positiveExamples.union(negativeExamples)
    //combining both labeledpoints
    trainingdata.cache()
    // Cache da3ta since Logistic Regression is an iterative algorithm.
    val Array(trainset, testset) = trainingdata.randomSplit(Array(0.7, 0.3))
    //splliting the dataset for training and testing using array randomsplit
    val model = NaiveBayes.train(trainset, 1.0)
    //creating naivebayes model in prder to train the test set
     val prediction = testset.map(x => (model.predict(x.features),x.label))
     //predicting the values of test set
     val accuracy = 1.0 * prediction.filter(x => x._1 == x._2).count()/testset.count()
     //finding the accuracy of the naive bayes model
     prediction.foreach(println)
     println(s"test accuracy = $accuracy. ")
     //printing accuracy of the model
    val lrLearner = new LogisticRegressionWithSGD()
      // Create a Logistic Regression learner which uses the LBFGS optimizer.
    val model2 = lrLearner.run(trainingdata)
        // Run the actual learning algorithm on the training data.
    val metrics = new MulticlassMetrics(prediction)
    val confusionMatrix = metrics.confusionMatrix
     println("Confusion Matrix= \n",confusionMatrix)
    val myModelStat=Seq(metrics.precision,metrics.fMeasure,metrics.recall)
    myModelStat.foreach(println(_))
        // Test on a positive example (bots) and a negative one (notbots).
        // First apply the same HashingTF feature transformation used on the training data.
    val posTestExample = tf.transform("Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)".split(""))
       // Now use the learned model to predict bots/notbots for new urls.
    println(s"Prediction for positive test example: ${model.predict(posTestExample)}")
    //testing the naive bayes model
    println(s"Prediction for positive test example: ${model2.predict(posTestExample)}")
    //testing logistic regression model on simple example
    sc.stop()
    //stop the spark context
}
}
