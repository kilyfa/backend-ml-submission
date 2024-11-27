const tf = require("@tensorflow/tfjs-node");
const path = require("path");

const loadModel = async () => {
  try {
    const model = await tf.loadGraphModel(`https://storage.googleapis.com/model_ml_submission1/model.json`);
    console.log("Model berhasil dimuat!");
    return model;
  } catch (error) {
    console.error("Gagal memuat model:", error);
    throw error;
  }
};

const predictImage = async (imagePath, model) => {
  try {
    const fs = require("fs");

    const imageBuffer = fs.readFileSync(imagePath);

    const tensor = tf.node.decodeImage(imageBuffer, 3).resizeNearestNeighbor([224, 224]).toFloat().expandDims();

    const prediction = model.predict(tensor);
    const score = await prediction.data();

    const value = Math.max(...score) * 100;

    let suggestion, result;

    if (value > 50) {
      result = "Cancer";
      suggestion = "Segera periksa ke dokter!";
    } else {
      result = "Non-cancer";
      suggestion = "Penyakit kanker tidak terdeteksi.";
    }

    return { suggestion, result, probability: value };
  } catch (error) {
    console.error("Gagal memproses gambar:", error);
    throw new Error("Terjadi kesalahan dalam melakukan prediksi");
  }
};

module.exports = { loadModel, predictImage };
