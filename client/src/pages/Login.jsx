import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import API from '../api/axios';
import { useAuth } from '../context/AuthContext';

export default function Login() {
    const [form, setForm] = useState({ email: '', password: '' });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        try {
            const { data } = await API.post('/auth/login', form);
            login(data);
            navigate('/dashboard');
        } catch (err) {
            setError(err.response?.data?.message || 'Invalid credentials');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-950 flex">
            {/* Left Panel */}
            <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-16 bg-slate-900 border-r border-slate-800">
                <div>
                    <h1 className="text-5xl text-white font-light leading-tight mb-6">
                        Intelligent<br />
                        <span className="text-amber-500">Credit Risk</span><br />
                        Assessment
                    </h1>
                    <p className="text-slate-400 text-lg font-light leading-relaxed max-w-md">
                        Advanced machine learning models to evaluate loan default probability with precision and confidence.
                    </p>
                </div>
                <div className="flex gap-12">
                    {[['98.2%', 'Accuracy'], ['5', 'ML Models'], ['Real-time', 'Predictions']].map(([val, label]) => (
                        <div key={label}>
                            <p className="text-2xl text-white font-light">{val}</p>
                            <p className="text-slate-500 text-xs tracking-widest uppercase mt-1">{label}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Right Panel */}
            <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
                <div className="w-full max-w-md">     
<h2 className="text-3xl text-amber-500 font-light mb-2 tracking-tight">Sign in</h2>                    <p className="text-slate-500 mb-10 text-sm">Access your risk assessment dashboard</p>

                    {error && (
                        <div className="border border-red-800 bg-red-950/50 text-red-400 px-4 py-3 rounded text-sm mb-6">
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-5">
                        <div>
                            <label className="text-slate-400 text-xs tracking-widest uppercase block mb-2">Email Address</label>
                            <input type="email" value={form.email}
                                onChange={(e) => setForm({ ...form, email: e.target.value })}
                                className="w-full bg-slate-900 text-white px-4 py-3 border border-slate-700 rounded focus:outline-none focus:border-amber-500 transition-colors text-sm"
                                placeholder="your@email.com" required />
                        </div>
                        <div>
                            <label className="text-slate-400 text-xs tracking-widest uppercase block mb-2">Password</label>
                            <input type="password" value={form.password}
                                onChange={(e) => setForm({ ...form, password: e.target.value })}
                                className="w-full bg-slate-900 text-white px-4 py-3 border border-slate-700 rounded focus:outline-none focus:border-amber-500 transition-colors text-sm"
                                placeholder="••••••••" required />
                        </div>
                        <button type="submit" disabled={loading}
                            className="w-full bg-amber-500 hover:bg-amber-400 text-slate-950 py-3 rounded font-medium tracking-wider text-sm transition-colors disabled:opacity-50">
                            {loading ? 'Signing in...' : 'Sign In'}
                        </button>
                    </form>
                    <p className="text-slate-600 text-sm mt-8 text-center">
                        No account?{' '}
                        <Link to="/register" className="text-amber-500 hover:text-amber-400 transition-colors">Create one</Link>
                    </p>
                </div>
            </div>
        </div>
    );
}