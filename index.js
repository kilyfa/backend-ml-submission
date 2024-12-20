const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { loadModel, predictImage } = require("./models/loadModel");
const db = require("./firebase");
const { v4: uuidv4 } = require("uuid");

const app = express();

const upload = multer({
  dest: "uploads/",
});

let model;
(async () => {
  try {
    model = await loadModel();
  } catch (error) {
    console.error("Failed to load model:", error);
    process.exit(1);
  }
})();

app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    const file = req.file;

    if (!file) {
      return res.status(400).json({
        status: "fail",
        message: "File not found.",
      });
    }

    if (file.size > 1000000) {
      fs.unlinkSync(file.path);
      return res.status(413).json({
        status: "fail",
        message: "Payload content length greater than maximum allowed: 1000000",
      });
    }

    const imageBuffer = fs.readFileSync(file.path);

    try {
      const tensor = tf.node.decodeImage(imageBuffer);

      if (tensor.shape[2] !== 3) {
        fs.unlinkSync(file.path);
        return res.status(400).json({
          status: "fail",
          message: "Terjadi kesalahan dalam melakukan prediksi",
        });
      }
    } catch (error) {
      console.error("Image validation error:", error.message);
      fs.unlinkSync(file.path);
      return res.status(400).json({
        status: "fail",
        message: "Terjadi kesalahan dalam melakukan prediksi",
      });
    }

    const imagePath = file.path;
    const result = await predictImage(imagePath, model);

    const predictionId = uuidv4();
    const predictionData = {
      id: predictionId,
      result: result.result,
      suggestion: result.suggestion,
      createdAt: new Date().toISOString(),
    };

    await db.collection("predictions").doc(predictionId).set(predictionData);

    fs.unlinkSync(imagePath);

    return res.status(201).json({
      status: "success",
      message: "Model is predicted successfully",
      data: predictionData,
    });
  } catch (error) {
    console.error("Prediction error:", error.message);
    return res.status(400).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});

app.get("/predict/histories", async (req, res) => {
  try {
    const predictionsRef = db.collection("predictions");
    const snapshot = await predictionsRef.get();

    if (snapshot.empty) {
      return res.status(200).json({
        status: "success",
        data: [],
      });
    }

    const histories = [];
    snapshot.forEach((doc) => {
      histories.push({
        id: doc.id,
        history: doc.data(),
      });
    });

    return res.status(200).json({
      status: "success",
      data: histories,
    });
  } catch (error) {
    console.error("Error fetching prediction history:", error.message);
    return res.status(500).json({
      status: "fail",
      message: "Error occurred while retrieving prediction history.",
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server is running at http://0.0.0.0:${PORT}`);
});
