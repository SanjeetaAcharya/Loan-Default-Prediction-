import { useAuth } from '../context/AuthContext';
import { Link } from 'react-router-dom';

export default function Dashboard() {
    const { user } = useAuth();

    const cards = [
        {
            title: 'Single Assessment',
            desc: 'Evaluate a single applicant by entering their financial profile to receive an instant default probability score.',
            link: '/predict',
            tag: 'Individual',
        },
        {
            title: 'Batch Processing',
            desc: 'Upload a CSV file to process multiple loan applications simultaneously and receive risk scores in bulk.',
            link: '/predict?tab=batch',
            tag: 'Bulk',
        },
        {
            title: 'Assessment History',
            desc: 'Review all previous predictions, track risk trends, and audit your assessment records.',
            link: '/history',
            tag: 'Records',
        },
    ];

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            <div className="max-w-7xl mx-auto px-8 py-16">

                {/* Header */}
                <div className="mb-16 border-b border-slate-800 pb-12">
                    <p className="text-amber-500 text-xs tracking-widest uppercase mb-3">Dashboard</p>
                    <h1 className="text-4xl font-light text-white mb-3">
                        Welcome back, <span className="text-amber-500">{user?.name}</span>
                    </h1>
                    <p className="text-slate-400 text-lg font-light">
                        Select an action to begin your credit risk assessment.
                    </p>
                </div>

                {/* Cards */}
<div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
    {cards.map((card) => (
        <Link key={card.title} to={card.link}
            className="group block bg-slate-900 border border-slate-800 hover:border-slate-700 rounded p-8 transition-all duration-300">
            <div className="flex justify-between items-center">
                <div>
                    <span className="text-xs tracking-widest uppercase text-amber-500/70 border border-amber-500/20 px-2 py-1 rounded-sm">
    {card.tag}
</span>
                    <h2 className="text-xl font-light text-white mt-4 mb-2">{card.title}</h2>
                    <p className="text-slate-500 text-sm">{card.desc}</p>
                </div>
                <span className="text-slate-600 group-hover:text-amber-500 transition-colors text-2xl ml-8">→</span>
            </div>
        </Link>
    ))}
</div>

                {/* Admin card */}
                {user?.role === 'admin' && (
                    <Link to="/admin"
                        className="group block bg-slate-900 border border-amber-500/20 hover:border-amber-500/50 rounded p-8 transition-all duration-300">
                        <div className="flex justify-between items-center">
                            <div>
                                <span className="text-xs tracking-widest uppercase text-amber-500/70 border border-amber-500/20 px-2 py-1 rounded-sm">
                                    Admin
                                </span>
                                <h2 className="text-xl font-light text-white mt-4 mb-2">System Administration</h2>
                                <p className="text-slate-500 text-sm">Manage users, monitor system activity and review all assessments.</p>
                            </div>
                            <span className="text-slate-600 group-hover:text-amber-500 transition-colors text-2xl ml-8">→</span>
                        </div>
                    </Link>
                )}
            </div>
        </div>
    );
}