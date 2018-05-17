import ml.dmlc.mxnet._
import ml.dmlc.mxnet.module.Module
import scala.collection.mutable.ArrayBuffer

val modelPrefix: String = "mx_mlp"
val modelEpoch: Int = 130

val mod = Module.loadCheckpoint(modelPrefix, modelEpoch, loadOptimizerStates = true)
//パラメータ確認
//val (argParams, auxParams) = mod.getParams

val dataiter = IO.ImageRecordIter(Map(
            "path_imgrec" -> "./test_v1.rec",
            "path_imglist" -> "./test_v1.lst",
            "data_shape" -> "(3,299,299)",
            "batch_size" -> "1"))

mod.bind(dataShapes = dataiter.provideData)
mod.initParams()

dataiter.reset()

var res = ArrayBuffer.empty[IndexedSeq[NDArray]]

while (dataiter.hasNext) {
    val dataBatch = dataiter.next()
    res += mod.predict(dataBatch)
}

val prob = res(0)
val arr = prob.toArray
