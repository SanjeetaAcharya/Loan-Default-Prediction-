import { useEffect, useState } from 'react';
import API from '../api/axios';

export default function History() {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        API.get('/predict/history')
            .then(({ data }) => setHistory(data))
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    if (loading) return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center">
            <p className="text-slate-500 text-sm tracking-widest uppercase">Loading records...</p>
        </div>
    );

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            <div className="max-w-7xl mx-auto px-8 py-16">

                <div className="mb-10 border-b border-slate-800 pb-10">
                    <p className="text-amber-500 text-xs tracking-widest uppercase mb-3">Records</p>
                    <h1 className="text-4xl font-light text-white">Assessment History</h1>
                </div>

                {history.length === 0 ? (
                    <div className="bg-slate-900 border border-slate-800 rounded p-12 text-center">
                        <p className="text-slate-500 text-sm">No assessments on record. Run your first prediction to get started.</p>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {history.map((item) => (
                            <div key={item._id} className="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded p-6 transition-colors">
                                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                                    <div className="flex items-center gap-4">
                                        <span className={`text-xs tracking-widest uppercase border px-2 py-1 rounded-sm ${
                                            item.type === 'batch'
                                                ? 'border-slate-600 text-slate-400'
                                                : 'border-slate-600 text-slate-400'
                                        }`}>
                                            {item.type === 'batch' ? 'Batch' : 'Individual'}
                                        </span>
                                        <span className="text-slate-500 text-xs">
                                            {new Date(item.createdAt).toLocaleDateString('en-US', {
                                                year: 'numeric', month: 'long', day: 'numeric',
                                                hour: '2-digit', minute: '2-digit'
                                            })}
                                        </span>
                                        <span className="text-slate-600 text-xs">
                                            {item.modelUsed?.replace(/_/g, ' ')}
                                        </span>
                                    </div>

                                    {item.type === 'single' && (
                                        <div className="flex items-center gap-4">
                                            <span className={`text-sm font-light ${
                                                item.risk === 'HIGH' ? 'text-red-400' : 'text-emerald-400'
                                            }`}>
                                                {(item.probability * 100).toFixed(1)}% probability
                                            </span>
                                            <span className={`text-xs tracking-widest uppercase border px-3 py-1 rounded-sm ${
                                                item.risk === 'HIGH'
                                                    ? 'border-red-800 text-red-400'
                                                    : 'border-emerald-800 text-emerald-400'
                                            }`}>
                                                {item.risk === 'HIGH' ? 'High Risk' : 'Low Risk'}
                                            </span>
                                        </div>
                                    )}

                                    {item.type === 'batch' && item.batchResults && (
                                        <div className="flex items-center gap-6 text-sm">
                                            <span className="text-slate-400">{item.batchResults.total} records</span>
                                            <span className="text-red-400">{item.batchResults.highRisk} high risk</span>
                                            <span className="text-emerald-400">{item.batchResults.lowRisk} low risk</span>
                                        </div>
                                    )}
                                </div>

                                {item.type === 'single' && item.riskFactors?.length > 0 && (
                                    <div className="mt-4 pt-4 border-t border-slate-800">
                                        <div className="flex flex-wrap gap-2">
                                            {item.riskFactors.map((r, i) => (
                                                <span key={i} className="text-xs text-red-400 bg-red-950/30 border border-red-900/50 px-3 py-1 rounded-sm">
                                                    {r}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}