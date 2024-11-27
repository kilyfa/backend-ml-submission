const tf = require("@tensorflow/tfjs-node");
const path = require("path");

const loadModel = async () => {
  try {
    const model = await tf.loadGraphModel(`file://models/model.json`);
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

    // Membaca file gambar
    const imageBuffer = fs.readFileSync(imagePath);

    // Decode gambar menjadi tensor dan lakukan preprocessing
    const tensor = tf.node
      .decodeImage(imageBuffer, 3) // Decode menjadi RGB tensor
      .resizeNearestNeighbor([224, 224]) // Sesuaikan ukuran ke [224, 224]
      .toFloat() // Konversi ke dtype float32 (wajib untuk model)
      .expandDims(); // Tambahkan dimensi batch

    // Prediksi menggunakan model
    const prediction = model.predict(tensor);
    const score = await prediction.data(); // Ambil hasil prediksi sebagai array

    // Ambil probabilitas maksimum dari hasil prediksi
    const value = Math.max(...score) * 100;

    // Logika prediksi
    let suggestion, result;

    if (value > 50) {
      result = "Cancer";
      suggestion = "Segera periksa ke dokter!";
    } else {
      result = "Non-cancer";
      suggestion = "Penyakit kanker tidak terdeteksi.";
    }

    // Kembalikan hasil prediksi
    return { suggestion, result, probability: value };
  } catch (error) {
    console.error("Gagal memproses gambar:", error);
    throw new Error("Terjadi kesalahan dalam melakukan prediksi");
  }
};

module.exports = { loadModel, predictImage };
