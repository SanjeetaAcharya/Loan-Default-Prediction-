import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import API from '../api/axios';
import { useAuth } from '../context/AuthContext';

export default function Register() {
    const [form, setForm] = useState({ name: '', email: '', password: '' });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        try {
            const { data } = await API.post('/auth/register', form);
            login(data);
            navigate('/dashboard');
        } catch (err) {
            setError(err.response?.data?.message || 'Registration failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-950 flex">
            <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-16 bg-slate-900 border-r border-slate-800">
                
                <div>
                    <h1 className="text-5xl text-white font-light leading-tight mb-6">
                        Start making<br />
                        <span className="text-amber-500">data-driven</span><br />
                        decisions
                    </h1>
                    <p className="text-slate-400 text-lg font-light leading-relaxed max-w-md">
                        Join financial institutions using our platform to assess credit risk with confidence.
                    </p>
                </div>
                <div className="border-t border-slate-800 pt-8">
                    <p className="text-slate-500 text-sm">Enterprise-grade security. Instant predictions.</p>
                </div>
            </div>

            <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
                <div className="w-full max-w-md">
                    
<h2 className="text-3xl text-amber-500 font-light mb-2 tracking-tight">Create account</h2>                    <p className="text-slate-500 mb-10 text-sm">Begin your risk assessment journey</p>

                    {error && (
                        <div className="border border-red-800 bg-red-950/50 text-red-400 px-4 py-3 rounded text-sm mb-6">
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-5">
                        {[
                            { label: 'Full Name', key: 'name', type: 'text', placeholder: 'Your Name' },
                            { label: 'Email Address', key: 'email', type: 'email', placeholder: 'your@email.com' },
                            { label: 'Password', key: 'password', type: 'password', placeholder: 'Minimum 6 characters' },
                        ].map(({ label, key, type, placeholder }) => (
                            <div key={key}>
                                <label className="text-slate-400 text-xs tracking-widest uppercase block mb-2">{label}</label>
                                <input type={type} value={form[key]}
                                    onChange={(e) => setForm({ ...form, [key]: e.target.value })}
                                    className="w-full bg-slate-900 text-white px-4 py-3 border border-slate-700 rounded focus:outline-none focus:border-amber-500 transition-colors text-sm"
                                    placeholder={placeholder} required />
                            </div>
                        ))}
                        <button type="submit" disabled={loading}
                            className="w-full bg-amber-500 hover:bg-amber-400 text-slate-950 py-3 rounded font-medium tracking-wider text-sm transition-colors disabled:opacity-50">
                            {loading ? 'Creating account...' : 'Create Account'}
                        </button>
                    </form>
                    <p className="text-slate-600 text-sm mt-8 text-center">
                        Already have an account?{' '}
                        <Link to="/login" className="text-amber-500 hover:text-amber-400 transition-colors">Sign in</Link>
                    </p>
                </div>
            </div>
        </div>
    );
}