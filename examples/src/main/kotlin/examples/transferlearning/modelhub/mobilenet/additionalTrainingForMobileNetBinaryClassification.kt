/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.mobilenet
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.freeze
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.saveModelConfiguration
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels.CV.Companion.createPreprocessing
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.embedded.dogsCatsSmallDatasetPath
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import org.jetbrains.kotlinx.dl.dataset.generator.FromFolders
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.predictLabel
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.predictProbabilities
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 8
private const val TEST_BATCH_SIZE = 16
private const val NUM_CLASSES = 2
private const val TRAIN_TEST_SPLIT_RATIO = 0.7

/**
 * This example demonstrates the transfer learning concept on MobileNet model:
 * - Model configuration, model weights and labels are obtained from [TFModelHub].
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - All layers, excluding the last [Dense], are added to the new Neural Network, its weights are frozen.
 * - New Dense layers are added and initialized via defined initializers.
 * - Model is re-trained on [dogsCatsSmallDatasetPath] dataset.
 * - Special preprocessing (used in MobileNet during training on ImageNet dataset) is applied to each image before prediction via [call] stage.
 *
 * We use the preprocessing DSL to describe the dataset generation pipeline.
 * We demonstrate the workflow on the subset of Kaggle Cats vs Dogs binary classification dataset.
 */
fun mobilenetWithAdditionalTrainingBinary() {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = TFModels.CV.MobileNetV2()
    val model = modelHub.loadModel(modelType)

    val hdfFile = modelHub.loadWeights(modelType)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.logSummary()
    }

    val layers = model.layers.toMutableList()
    layers.forEach(Layer::freeze)

    val lastLayer = layers.last()
    for (outboundLayer in lastLayer.inboundLayers)
        outboundLayer.outboundLayers.remove(lastLayer)

    layers.removeLast()

    var x = Dense(
        name = "top_dense",
        kernelInitializer = GlorotUniform(),
        biasInitializer = GlorotUniform(),
        outputSize = 200,
        activation = Activations.Relu
    )(layers.last())

    x = Dense(
        name = "pred",
        kernelInitializer = GlorotUniform(),
        biasInitializer = GlorotUniform(),
        outputSize = 2,
        activation = Activations.Linear,
    )(x)

    val model2 = Functional.fromOutput(x)

    val dogsCatsImages = dogsCatsSmallDatasetPath() + "/cat"
    val dataset = OnHeapDataset.create(
        File(dogsCatsImages),
        FromFolders(mapping = mapOf("cat" to 1)),
        //FromFolders(mapping = mapOf("cat" to 0, "dog" to 1)),
        modelType.createPreprocessing(model2)
    ).shuffle()
    val (train, test) = dataset.split(TRAIN_TEST_SPLIT_RATIO)

    model2.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        it.loadWeightsForFrozenLayers(hdfFile)
        val accuracyBeforeTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBeforeTraining")

        it.fit(
            dataset = train,
            batchSize = TRAINING_BATCH_SIZE,
            epochs = EPOCHS
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")



        val model2Dir = File("my-models/mobilenet_with_additional_training")
        it.save(model2Dir,
            saveOptimizerState = false,
            savingFormat = SavingFormat.JsonConfigCustomVariables(),//SavingFormat.JsonConfigCustomVariables(),
            writingMode = WritingMode.OVERRIDE)
        //it.saveModelConfiguration("$model2Dir/modelConfig.json", isKerasFullyCompatible = true)

//        val model = Functional.loadModelConfiguration(File("my-models/mobilenet_with_additional_training/modelConfig.json"))
//        model.use {
//            setUpModel(it)
//            File(dogsCatsSmallDatasetPath() + "/cat").listFiles().forEach { imaFile ->
//                val inputData = myFileDataLoader.load(imaFile)
//                val res = it.predictProbabilities(inputData)
//                println("Predicted object for ${imaFile.name} is ${res[0]}")
//            }
//        }


    }
}

/** */
fun main(): Unit {
    mobilenetWithAdditionalTrainingBinary()



    val model = Functional.loadModelConfiguration(File("my-models/mobilenet_with_additional_training/modelConfig.json"))
    model.use {
        setUpModel(it)
        File(dogsCatsSmallDatasetPath() + "/cat").listFiles().forEach { imaFile ->
            val inputData = myFileDataLoader.load(imaFile)
            val res = it.predictLabel(inputData)
            println("Predicted object for ${imaFile.name} is ${res}")
        }
        File(dogsCatsSmallDatasetPath() + "/dog").listFiles().forEach { imaFile ->
            val inputData = myFileDataLoader.load(imaFile)
            val res = it.predictLabel(inputData)
            println("Predicted object for ${imaFile.name} is ${res}")
        }
    }
}


