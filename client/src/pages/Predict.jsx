import { useState } from 'react';
import API from '../api/axios';

const MODELS = [
    { value: 'random_forest', label: 'Random Forest' },
    { value: 'logistic_regression', label: 'Logistic Regression' },
    { value: 'decision_tree', label: 'Decision Tree' },
    { value: 'knn', label: 'K-Nearest Neighbors' },
    { value: 'naive_bayes', label: 'Naive Bayes' },
];

const DEFAULT_FORM = {
    model: 'random_forest',
    credit_score: 700, annual_income: 60000, monthly_debt: 1000,
    tax_liens: 0, bankruptcies: 0, credit_problems: 0,
    years_in_job: 3, open_accounts: 8, credit_history: 10,
    max_open_credit: 100000, loan_amount: 50000, credit_balance: 30000,
    months_delinquent: 0, home_ownership: 'Own Home',
    purpose: 'debt consolidation', term: 'Short Term',
};

const Field = ({ label, name, value, onChange, type = 'number' }) => (
    <div>
        <label className="text-slate-500 text-xs tracking-widest uppercase block mb-2">{label}</label>
        <input type={type} name={name} value={value} onChange={onChange}
            className="w-full bg-slate-950 text-white px-3 py-2.5 border border-slate-700 rounded focus:outline-none focus:border-amber-500 transition-colors text-sm" />
    </div>
);

const SelectField = ({ label, name, value, onChange, options }) => (
    <div>
        <label className="text-slate-500 text-xs tracking-widest uppercase block mb-2">{label}</label>
        <select name={name} value={value} onChange={onChange}
            className="w-full bg-slate-950 text-white px-3 py-2.5 border border-slate-700 rounded focus:outline-none focus:border-amber-500 transition-colors text-sm">
            {options.map(o => <option key={o.value ?? o} value={o.value ?? o}>{o.label ?? o}</option>)}
        </select>
    </div>
);

export default function Predict() {
    const [tab, setTab] = useState('single');
    const [form, setForm] = useState(DEFAULT_FORM);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [file, setFile] = useState(null);
    const [batchResult, setBatchResult] = useState(null);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setForm({ ...form, [name]: isNaN(value) || value === '' ? value : Number(value) });
    };

    const handleSinglePredict = async (e) => {
        e.preventDefault();
        setLoading(true); setError(''); setResult(null);
        try {
            const { data } = await API.post('/predict/single', form);
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.message || 'Prediction failed');
        } finally { setLoading(false); }
    };

    const handleBatchPredict = async (e) => {
        e.preventDefault();
        if (!file) return setError('Please select a CSV file');
        setLoading(true); setError(''); setBatchResult(null);
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('model', form.model);
            const { data } = await API.post('/predict/batch', formData);
            setBatchResult(data);
        } catch (err) {
            setError(err.response?.data?.message || 'Batch prediction failed');
        } finally { setLoading(false); }
    };

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            <div className="max-w-7xl mx-auto px-8 py-16">

                {/* Header */}
                <div className="mb-10 border-b border-slate-800 pb-10">
                    <p className="text-amber-500 text-xs tracking-widest uppercase mb-3">Risk Assessment</p>
                    <h1 className="text-4xl font-light text-white">Loan Default Prediction</h1>
                </div>

                {/* Model + Tab row */}
                <div className="flex flex-col md:flex-row justify-between gap-6 mb-10">
                    <div className="flex gap-1 bg-slate-900 border border-slate-800 rounded p-1">
                        {['single', 'batch'].map((t) => (
                            <button key={t} onClick={() => setTab(t)}
                                className={`px-6 py-2 rounded text-sm tracking-wider transition-colors ${
                                    tab === t ? 'bg-amber-500 text-slate-950 font-medium' : 'text-slate-400 hover:text-white'
                                }`}>
                                {t === 'single' ? 'Individual' : 'Batch CSV'}
                            </button>
                        ))}
                    </div>
                    <div className="w-full md:w-72">
                        <SelectField label="Model" name="model" value={form.model} onChange={handleChange} options={MODELS} />
                    </div>
                </div>

                {error && (
                    <div className="border border-red-800 bg-red-950/50 text-red-400 px-4 py-3 rounded text-sm mb-8">
                        {error}
                    </div>
                )}

                {/* Single Prediction */}
                {tab === 'single' && (
                    <form onSubmit={handleSinglePredict}>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                            <div className="bg-slate-900 border border-slate-800 rounded p-6 space-y-4">
                                <h3 className="text-xs tracking-widest uppercase text-slate-400 border-b border-slate-800 pb-3 mb-4">Credit Profile</h3>
                                <Field label="Credit Score (300–850)" name="credit_score" value={form.credit_score} onChange={handleChange} />
                                <Field label="Credit Problems" name="credit_problems" value={form.credit_problems} onChange={handleChange} />
                                <Field label="Years of Credit History" name="credit_history" value={form.credit_history} onChange={handleChange} />
                                <Field label="Open Accounts" name="open_accounts" value={form.open_accounts} onChange={handleChange} />
                                <Field label="Maximum Open Credit ($)" name="max_open_credit" value={form.max_open_credit} onChange={handleChange} />
                                <Field label="Current Credit Balance ($)" name="credit_balance" value={form.credit_balance} onChange={handleChange} />
                                <Field label="Months Since Delinquent" name="months_delinquent" value={form.months_delinquent} onChange={handleChange} />
                            </div>

                            <div className="bg-slate-900 border border-slate-800 rounded p-6 space-y-4">
                                <h3 className="text-xs tracking-widest uppercase text-slate-400 border-b border-slate-800 pb-3 mb-4">Financial Profile</h3>
                                <Field label="Annual Income ($)" name="annual_income" value={form.annual_income} onChange={handleChange} />
                                <Field label="Monthly Debt ($)" name="monthly_debt" value={form.monthly_debt} onChange={handleChange} />
                                <Field label="Loan Amount ($)" name="loan_amount" value={form.loan_amount} onChange={handleChange} />
                                <Field label="Tax Liens" name="tax_liens" value={form.tax_liens} onChange={handleChange} />
                                <Field label="Bankruptcies" name="bankruptcies" value={form.bankruptcies} onChange={handleChange} />
                            </div>

                            <div className="bg-slate-900 border border-slate-800 rounded p-6 space-y-4">
                                <h3 className="text-xs tracking-widest uppercase text-slate-400 border-b border-slate-800 pb-3 mb-4">Loan Details</h3>
                                <Field label="Years in Current Job" name="years_in_job" value={form.years_in_job} onChange={handleChange} />
                                <SelectField label="Home Ownership" name="home_ownership" value={form.home_ownership} onChange={handleChange}
                                    options={['Have Mortgage', 'Home Mortgage', 'Own Home', 'Rent']} />
                                <SelectField label="Loan Purpose" name="purpose" value={form.purpose} onChange={handleChange}
                                    options={['business loan','buy a car','buy house','debt consolidation','educational expenses','home improvements','major purchase','medical bills','moving','other','small business','take a trip','vacation','wedding']} />
                                <SelectField label="Loan Term" name="term" value={form.term} onChange={handleChange}
                                    options={['Short Term', 'Long Term']} />
                            </div>
                        </div>

                        <button type="submit" disabled={loading}
                            className="w-full bg-amber-500 hover:bg-amber-400 text-slate-950 py-3.5 rounded font-medium tracking-widest text-sm uppercase transition-colors disabled:opacity-50">
                            {loading ? 'Analysing...' : 'Run Assessment'}
                        </button>
                    </form>
                )}

                {/* Batch */}
                {tab === 'batch' && (
                    <form onSubmit={handleBatchPredict}>
                        <div className="bg-slate-900 border border-slate-800 rounded p-8 mb-8">
                            <h3 className="text-xs tracking-widest uppercase text-slate-400 border-b border-slate-800 pb-3 mb-6">Upload CSV File</h3>
                            <p className="text-slate-500 text-sm mb-6 leading-relaxed">
                                Required columns: Credit Score, Annual Income, Monthly Debt, Tax Liens, Bankruptcies,
                                Number of Credit Problems, Years in current job, Number of Open Accounts,
                                Years of Credit History, Maximum Open Credit, Current Loan Amount,
                                Current Credit Balance, Months since last delinquent, Home Ownership, Purpose, Term
                            </p>
                            <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])}
                                className="w-full bg-slate-950 text-slate-300 px-4 py-3 border border-slate-700 rounded text-sm mb-4 file:mr-4 file:py-1 file:px-4 file:rounded file:border-0 file:text-xs file:bg-amber-500 file:text-slate-950 file:tracking-wider" />
                            <button type="submit" disabled={loading}
                                className="w-full bg-amber-500 hover:bg-amber-400 text-slate-950 py-3.5 rounded font-medium tracking-widest text-sm uppercase transition-colors disabled:opacity-50">
                                {loading ? 'Processing...' : 'Process Batch'}
                            </button>
                        </div>
                    </form>
                )}

                {/* Single Result */}
                {result && (
                    <div className={`mt-8 border rounded p-8 ${
                        result.risk === 'HIGH' ? 'border-red-800 bg-red-950/20' : 'border-emerald-800 bg-emerald-950/20'
                    }`}>
                        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-6 pb-6 border-b border-slate-800">
                            <div>
                                <p className="text-xs tracking-widest uppercase text-slate-500 mb-2">Assessment Result</p>
                                <p className={`text-4xl font-light ${result.risk === 'HIGH' ? 'text-red-400' : 'text-emerald-400'}`}>
                                    {(result.probability * 100).toFixed(1)}%
                                </p>
                                <p className="text-slate-400 text-sm mt-1">Default Probability</p>
                            </div>
                            <div className={`px-6 py-3 rounded border text-sm tracking-widest uppercase font-medium ${
                                result.risk === 'HIGH'
                                    ? 'border-red-700 text-red-400 bg-red-950/30'
                                    : 'border-emerald-700 text-emerald-400 bg-emerald-950/30'
                            }`}>
                                {result.risk === 'HIGH' ? 'High Risk' : 'Low Risk'}
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <p className="text-xs tracking-widest uppercase text-slate-500 mb-3">Risk Indicators</p>
                                {result.risk_factors?.length > 0 ? result.risk_factors.map((r, i) => (
                                    <p key={i} className="text-red-400 text-sm py-1.5 border-b border-red-900/30 last:border-0">{r}</p>
                                )) : (
                                    <p className="text-emerald-400 text-sm">No significant risk factors identified.</p>
                                )}
                            </div>
                            <div>
                                <p className="text-xs tracking-widest uppercase text-slate-500 mb-3">Key Metrics</p>
                                <div className="space-y-2">
                                    <div className="flex justify-between text-sm py-1.5 border-b border-slate-800">
                                        <span className="text-slate-400">Debt-to-Income Ratio</span>
                                        <span className={result.dti_ratio > 0.5 ? 'text-red-400' : 'text-emerald-400'}>
                                            {(result.dti_ratio * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="flex justify-between text-sm py-1.5 border-b border-slate-800">
                                        <span className="text-slate-400">Model Used</span>
                                        <span className="text-white">{result.model_used?.replace(/_/g, ' ')}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Batch Result */}
                {batchResult && (
                    <div className="mt-8 bg-slate-900 border border-slate-800 rounded p-8">
                        <p className="text-xs tracking-widest uppercase text-slate-500 mb-6">Batch Assessment Results</p>
                        <div className="grid grid-cols-3 gap-4 mb-8">
                            {[
                                { label: 'Total Processed', value: batchResult.total, color: 'text-white' },
                                { label: 'High Risk', value: batchResult.high_risk, color: 'text-red-400' },
                                { label: 'Low Risk', value: batchResult.low_risk, color: 'text-emerald-400' },
                            ].map(({ label, value, color }) => (
                                <div key={label} className="bg-slate-950 border border-slate-800 rounded p-5 text-center">
                                    <p className={`text-3xl font-light ${color}`}>{value}</p>
                                    <p className="text-slate-500 text-xs tracking-wider uppercase mt-2">{label}</p>
                                </div>
                            ))}
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-slate-800">
                                        <th className="py-3 text-left text-xs tracking-widest uppercase text-slate-500">Row</th>
                                        <th className="py-3 text-left text-xs tracking-widest uppercase text-slate-500">Risk Level</th>
                                        <th className="py-3 text-left text-xs tracking-widest uppercase text-slate-500">Probability</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {batchResult.results.slice(0, 20).map((r) => (
                                        <tr key={r.row} className="border-b border-slate-900">
                                            <td className="py-3 text-slate-400">{r.row}</td>
                                            <td className={`py-3 font-medium ${r.risk === 'HIGH' ? 'text-red-400' : 'text-emerald-400'}`}>{r.risk}</td>
                                            <td className="py-3 text-slate-300">{r.probability ? (r.probability * 100).toFixed(1) + '%' : '—'}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                            {batchResult.results.length > 20 && (
                                <p className="text-slate-500 text-xs mt-4">Showing 20 of {batchResult.results.length} records</p>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}