import { useEffect, useState } from 'react';
import API from '../api/axios';

export default function Admin() {
    const [stats, setStats] = useState(null);
    const [users, setUsers] = useState([]);
    const [tab, setTab] = useState('stats');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        Promise.all([API.get('/admin/stats'), API.get('/admin/users')])
            .then(([s, u]) => { setStats(s.data); setUsers(u.data); })
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    const handleDelete = async (id) => {
        if (!confirm('Delete this user and all their data?')) return;
        try {
            await API.delete(`/admin/users/${id}`);
            setUsers(users.filter(u => u._id !== id));
        } catch (err) {
            alert(err.response?.data?.message || 'Failed');
        }
    };

    if (loading) return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center">
            <p className="text-slate-500 text-sm tracking-widest uppercase">Loading...</p>
        </div>
    );

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            <div className="max-w-7xl mx-auto px-8 py-16">

                <div className="mb-10 border-b border-slate-800 pb-10">
                    <p className="text-amber-500 text-xs tracking-widest uppercase mb-3">Administration</p>
                    <h1 className="text-4xl font-light text-white">System Dashboard</h1>
                </div>

                {/* Tabs */}
                <div className="flex gap-1 bg-slate-900 border border-slate-800 rounded p-1 w-fit mb-10">
                    {[['stats', 'Overview'], ['users', 'Users']].map(([val, label]) => (
                        <button key={val} onClick={() => setTab(val)}
                            className={`px-6 py-2 rounded text-sm tracking-wider transition-colors ${
                                tab === val ? 'bg-amber-500 text-slate-950 font-medium' : 'text-slate-400 hover:text-white'
                            }`}>
                            {label}
                        </button>
                    ))}
                </div>

                {/* Stats */}
                {tab === 'stats' && stats && (
                    <div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
                            {[
                                { label: 'Total Users', value: stats.totalUsers },
                                { label: 'Total Predictions', value: stats.totalPredictions },
                                { label: 'High Risk', value: stats.highRiskCount },
                                { label: 'Low Risk', value: stats.lowRiskCount },
                            ].map(({ label, value }) => (
                                <div key={label} className="bg-slate-900 border border-slate-800 rounded p-6">
                                    <p className="text-3xl font-light text-white">{value}</p>
                                    <p className="text-slate-500 text-xs tracking-widest uppercase mt-2">{label}</p>
                                </div>
                            ))}
                        </div>

                        <div className="bg-slate-900 border border-slate-800 rounded p-6">
                            <p className="text-xs tracking-widest uppercase text-slate-500 border-b border-slate-800 pb-4 mb-6">
                                Activity — Last 7 Days
                            </p>
                            {stats.dailyStats.length === 0 ? (
                                <p className="text-slate-500 text-sm">No activity in the last 7 days.</p>
                            ) : (
                                <div className="space-y-3">
                                    {stats.dailyStats.map((d) => (
                                        <div key={d._id} className="flex items-center gap-6">
                                            <span className="text-slate-500 text-xs w-28 font-mono">{d._id}</span>
                                            <div className="flex-1 bg-slate-800 rounded-full h-1.5">
                                                <div className="bg-amber-500 h-1.5 rounded-full transition-all"
                                                    style={{ width: `${Math.min((d.count / 20) * 100, 100)}%` }} />
                                            </div>
                                            <span className="text-slate-400 text-sm w-6 text-right">{d.count}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Users */}
                {tab === 'users' && (
                    <div className="bg-slate-900 border border-slate-800 rounded">
                        <div className="px-6 py-4 border-b border-slate-800">
                            <p className="text-xs tracking-widest uppercase text-slate-500">
                                {users.length} registered users
                            </p>
                        </div>
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-slate-800">
                                    {['Name', 'Email', 'Role', 'Joined', ''].map(h => (
                                        <th key={h} className="px-6 py-4 text-left text-xs tracking-widest uppercase text-slate-500">{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {users.map((u) => (
                                    <tr key={u._id} className="border-b border-slate-900 hover:bg-slate-800/50 transition-colors">
                                        <td className="px-6 py-4 text-white">{u.name}</td>
                                        <td className="px-6 py-4 text-slate-400">{u.email}</td>
                                        <td className="px-6 py-4">
                                            <span className={`text-xs tracking-widest uppercase border px-2 py-1 rounded-sm ${
                                                u.role === 'admin'
                                                    ? 'border-amber-600 text-amber-500'
                                                    : 'border-slate-700 text-slate-400'
                                            }`}>
                                                {u.role}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-slate-500 text-xs font-mono">
                                            {new Date(u.createdAt).toLocaleDateString()}
                                        </td>
                                        <td className="px-6 py-4">
                                            {u.role !== 'admin' && (
                                                <button onClick={() => handleDelete(u._id)}
                                                    className="text-xs tracking-wider text-slate-500 hover:text-red-400 transition-colors uppercase">
                                                    Remove
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}