const express    = require('express');
const axios      = require('axios');
const multer     = require('multer');
const FormData   = require('form-data');
const Prediction = require('../models/Prediction');
const { protect } = require('../middleware/auth');

const router  = express.Router();
const upload  = multer({ storage: multer.memoryStorage() });
const ML_API  = process.env.ML_API_URL || 'http://localhost:8000';

// ── POST /api/predict/single ──────────────────────────────────
router.post('/single', protect, async (req, res) => {
    try {
        // Forward to FastAPI
        const { data } = await axios.post(`${ML_API}/predict`, req.body);

        // Save to MongoDB
        const prediction = await Prediction.create({
            user:        req.user._id,
            type:        'single',
            inputData:   req.body,
            prediction:  data.prediction,
            probability: data.probability,
            risk:        data.risk,
            riskFactors: data.risk_factors,
            dtiRatio:    data.dti_ratio,
            modelUsed:   data.model_used,
        });

        res.json({ ...data, _id: prediction._id });

    } catch (err) {
        const message = err.response?.data?.detail || err.message;
        res.status(500).json({ message });
    }
});

// ── POST /api/predict/batch ───────────────────────────────────
router.post('/batch', protect, upload.single('file'), async (req, res) => {
    if (!req.file)
        return res.status(400).json({ message: 'No file uploaded' });

    try {
        // Forward file to FastAPI
        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename:    req.file.originalname,
            contentType: req.file.mimetype,
        });
        formData.append('model', req.body.model || 'random_forest');

        const { data } = await axios.post(`${ML_API}/predict/batch`, formData, {
            headers: formData.getHeaders(),
        });

        // Save to MongoDB
        const prediction = await Prediction.create({
            user:  req.user._id,
            type:  'batch',
            modelUsed: req.body.model || 'random_forest',
            batchResults: {
                total:    data.total,
                highRisk: data.high_risk,
                lowRisk:  data.low_risk,
                results:  data.results,
            },
        });

        res.json({ ...data, _id: prediction._id });

    } catch (err) {
        const message = err.response?.data?.detail || err.message;
        res.status(500).json({ message });
    }
});

// ── GET /api/predict/history ──────────────────────────────────
router.get('/history', protect, async (req, res) => {
    try {
        const predictions = await Prediction.find({ user: req.user._id })
            .sort({ createdAt: -1 })
            .limit(50);
        res.json(predictions);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

// ── GET /api/predict/models ───────────────────────────────────
router.get('/models', protect, async (req, res) => {
    try {
        const { data } = await axios.get(`${ML_API}/models`);
        res.json(data);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

module.exports = router;