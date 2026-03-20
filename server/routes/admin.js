const express    = require('express');
const User       = require('../models/User');
const Prediction = require('../models/Prediction');
const { protect, adminOnly } = require('../middleware/auth');

const router = express.Router();

// All admin routes require login + admin role
router.use(protect, adminOnly);

// ── GET /api/admin/stats ──────────────────────────────────────
router.get('/stats', async (req, res) => {
    try {
        const totalUsers       = await User.countDocuments();
        const totalPredictions = await Prediction.countDocuments();
        const highRiskCount    = await Prediction.countDocuments({ risk: 'HIGH', type: 'single' });
        const lowRiskCount     = await Prediction.countDocuments({ risk: 'LOW',  type: 'single' });
        const batchCount       = await Prediction.countDocuments({ type: 'batch' });

        // Predictions per day (last 7 days)
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

        const dailyStats = await Prediction.aggregate([
            { $match: { createdAt: { $gte: sevenDaysAgo }, type: 'single' } },
            {
                $group: {
                    _id:   { $dateToString: { format: '%Y-%m-%d', date: '$createdAt' } },
                    count: { $sum: 1 },
                }
            },
            { $sort: { _id: 1 } }
        ]);

        res.json({
            totalUsers,
            totalPredictions,
            highRiskCount,
            lowRiskCount,
            batchCount,
            dailyStats,
        });
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

// ── GET /api/admin/users ──────────────────────────────────────
router.get('/users', async (req, res) => {
    try {
        const users = await User.find().select('-password').sort({ createdAt: -1 });
        res.json(users);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

// ── DELETE /api/admin/users/:id ───────────────────────────────
router.delete('/users/:id', async (req, res) => {
    try {
        const user = await User.findById(req.params.id);
        if (!user) return res.status(404).json({ message: 'User not found' });
        if (user.role === 'admin') return res.status(400).json({ message: 'Cannot delete admin' });

        await user.deleteOne();
        await Prediction.deleteMany({ user: req.params.id });

        res.json({ message: 'User and their predictions deleted' });
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

// ── GET /api/admin/predictions ────────────────────────────────
router.get('/predictions', async (req, res) => {
    try {
        const predictions = await Prediction.find()
            .populate('user', 'name email')
            .sort({ createdAt: -1 })
            .limit(100);
        res.json(predictions);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

module.exports = router;