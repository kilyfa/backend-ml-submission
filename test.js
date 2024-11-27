const tf = require("@tensorflow/tfjs-node");

(async () => {
  try {
    // Coba muat TensorFlow.js dan tampilkan versi
    console.log("TensorFlow.js berhasil dimuat!");
    console.log("Versi TensorFlow.js:", tf.version.tfjs);

    // Buat tensor sederhana untuk menguji fungsionalitas
    const tensor = tf.tensor([1, 2, 3, 4]);
    console.log("Tensor:", tensor.toString());
  } catch (error) {
    console.error("Gagal memuat TensorFlow.js:", error);
  }
})();
