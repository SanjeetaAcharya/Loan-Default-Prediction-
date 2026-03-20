import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Navbar() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    const navLinks = [
        { to: '/dashboard', label: 'Overview' },
        { to: '/predict', label: 'Predict' },
        { to: '/history', label: 'History' },
        ...(user?.role === 'admin' ? [{ to: '/admin', label: 'Admin' }] : []),
    ];

    return (
        <nav className="bg-slate-950 border-b border-slate-800 px-8 py-4">
            <div className="max-w-7xl mx-auto flex justify-between items-center">
                <Link to="/dashboard" className="flex items-center gap-3">
                    <svg width="30" height="30" viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="30" height="30" rx="4" fill="#0f172a" stroke="#f59e0b" strokeWidth="1.5"/>
    {/* L */}
    <line x1="7" y1="7" x2="7" y2="22" stroke="#f59e0b" strokeWidth="2.2" strokeLinecap="round"/>
    <line x1="7" y1="22" x2="13" y2="22" stroke="#f59e0b" strokeWidth="2.2" strokeLinecap="round"/>
    {/* P */}
    <line x1="17" y1="7" x2="17" y2="22" stroke="#f59e0b" strokeWidth="2.2" strokeLinecap="round"/>
    <path d="M17 7 Q24 7 24 12 Q24 17 17 17" stroke="#f59e0b" strokeWidth="2.2" strokeLinecap="round" fill="none"/>
</svg>
                    <span className="text-white text-base tracking-widest uppercase font-light">LoanPredict</span>
                </Link>

                {user && (
                    <div className="flex items-center gap-8">
                        <div className="flex items-center gap-6">
                            {navLinks.map(({ to, label }) => (
                                <Link key={to} to={to}
                                    className={`text-sm tracking-wider transition-colors ${
                                        location.pathname === to
                                            ? 'text-amber-500'
                                            : 'text-slate-400 hover:text-white'
                                    }`}>
                                    {label}
                                </Link>
                            ))}
                        </div>
                        <div className="flex items-center gap-4 border-l border-slate-800 pl-8">
                            <span className="text-slate-500 text-xs tracking-wider">{user.name}</span>
                            <button onClick={handleLogout}
className="text-xs tracking-widest uppercase text-red-500 hover:text-red-400 transition-colors">                                Sign out
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </nav>
    );
}