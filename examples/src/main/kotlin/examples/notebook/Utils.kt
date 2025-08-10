package examples.notebook

import java.io.File
import java.nio.file.Files
import java.util.*

val mobileNetV2Labels = File("/Users/urs/development/github/ai/kotlindl/notebook/mobilenetv2-labels.txt").readLines().map { it.replace(Regex("""^[0-9\s]*"""), "") }.withIndex().associate { it.index to it.value }
fun encodeImageToBase64(imagePath: String): String {
    val file = File(imagePath)
    val bytes = Files.readAllBytes(file.toPath())
    return Base64.getEncoder().encodeToString(bytes)
}

fun File.htmlRowPrediction(classPrediction:Int, vectorPrediction:FloatArray, labels:Map<Int, String> = mobileNetV2Labels): String {
    val img =  """<img src="data:image/jpeg;base64,${encodeImageToBase64(absolutePath)}" width="100" height="100"/>"""
    return """<tr><td>$img</td><td><b>${labels[classPrediction]}</b></td><td>Accuracy: ${vectorPrediction.max()}</td></tr>"""
}
