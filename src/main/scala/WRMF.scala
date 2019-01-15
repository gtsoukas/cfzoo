// TODOs:
//  * Issue: some users don't get rankings

import java.io._

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

case class Rating(user: Int, item: Int, rating: Float)

object WRMF {

  def main(args: Array[String]) {

    val RANK = 32
    val N_ITER = 3

    if (args.length != 4) {
      println("Usage: need exactly four parameters")
      sys.exit(1)
    }

    val spark = SparkSession
      .builder
      .appName("WRMF from wrmfzoo")
      .getOrCreate()

    val TRAIN_FILE = args(0)  // "data/lastfm/train.svm"
    val TEST_FILE = args(1)   //"data/lastfm/test.svm"
    val NEGATIVES_FILE = args(2) //"data/lastfm/negatives.svm"
    val RANGINGS_FILE = args(3) //"data/lastfm/ranking_sparkml"

    def parseLibSVM(d:RDD[String]):RDD[Rating] =
      d.flatMap(r => {
        val a = r.split(" ")
        a.tail.map(b => {
          val c = b.split(":")
          Rating(a.head.toInt, c(0).toInt, c(1).toFloat)
        })
      })

    import spark.implicits._
    val train = parseLibSVM(
      spark.read.text(TRAIN_FILE).rdd.map(x => x.getAs[String]("value"))).toDF
    val test = parseLibSVM(
      spark.read.text(TEST_FILE).rdd.map(x => x.getAs[String]("value"))).toDF
    val negatives = parseLibSVM(
      spark.read.text(NEGATIVES_FILE).rdd.map(x => x.getAs[String]("value")))
      .toDF

    train.persist(StorageLevel.MEMORY_ONLY).count

    val als = new ALS()
      .setMaxIter(N_ITER)
      .setRegParam(0.0)
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .setImplicitPrefs(true)
      .setRank(RANK)
      .setNumBlocks(8)
      .setCheckpointInterval(-1)
    //.setAlpha(128.0)

    println("learning model")
    val t1 = System.currentTimeMillis()
    val model = als.fit(train)
    println("... took " + (System.currentTimeMillis() - t1) / 1000.0 + " s")

    println("writing rankings to single file " + RANGINGS_FILE)
    val t2 = System.currentTimeMillis()
    val d0 = test.union(negatives)
    // model.setColdStartStrategy("nan")
    val d1 = model.transform(d0)

    val wSpec2 = Window.partitionBy("user").orderBy(desc("prediction"))

    //TODO investigate handling of NaNs
    val d2 = d1
      .filter(! $"prediction".isNaN)
      .withColumn("rank", row_number().over(wSpec2))
      .filter($"rank" <= 10)
      .rdd
        .map(x => (x.getAs[Int]("user"), (x.getAs[Int]("item"), x.getAs[Float]("prediction"))))
        .groupByKey()
        .map(x => x._1.toString + ", [" + x._2.toList.sortBy(-_._2).mkString(", ") + "]")

    // Dont't do this for large datasets
    val rankings_file = new File(RANGINGS_FILE)
    val bw = new BufferedWriter(new FileWriter(rankings_file))
    d2.collect.foreach(x => {
      bw.write(x)
      bw.write("\n")
    })
    bw.close()
    println("... took " + (System.currentTimeMillis() - t2) / 1000.0 + " s")

    spark.stop()
  }
}
