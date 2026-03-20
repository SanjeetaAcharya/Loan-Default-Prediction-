const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref:  'User',
        required: true,
    },
    type: {
        type: String,
        enum: ['single', 'batch'],
        default: 'single',
    },

    // Input data (single prediction)
    inputData: {
        type: Object,
    },

    // Result (single prediction)
    prediction:  Number,
    probability: Number,
    risk:        String,
    riskFactors: [String],
    dtiRatio:    Number,
    modelUsed:   String,

    // Batch results
    batchResults: {
        total:     Number,
        highRisk:  Number,
        lowRisk:   Number,
        results:   Array,
    },

}, { timestamps: true });

module.exports = mongoose.model('Prediction', predictionSchema);