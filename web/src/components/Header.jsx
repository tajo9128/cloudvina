import { Link, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'
import { ChevronDown } from 'lucide-react'

export default function Header() {
    const [scrolled, setScrolled] = useState(false)
    const [isAdmin, setIsAdmin] = useState(false)
    const [user, setUser] = useState(null)
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
    const location = useLocation()

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20)
        }

        checkAuth()

        window.addEventListener('scroll', handleScroll)
        return () => window.removeEventListener('scroll', handleScroll)
    }, [])

    const checkAuth = async () => {
        const { data: { session } } = await supabase.auth.getSession()
        if (session?.user) {
            setUser(session.user)
            const { data } = await supabase
                .from('profiles')
                .select('is_admin')
                .eq('id', session.user.id)
                .single()

            if (data?.is_admin) setIsAdmin(true)
        }
    }

    // Navigation Configuration
    const navConfig = [
        { name: 'Home', path: '/' },
        {
            name: 'Features',
            path: '#',
            dropdown: [
                { name: 'Molecular Docking', path: '/molecular-docking-online' },
                { name: '3D Viewer', path: '/3d-viewer' },
                { name: 'MD Simulation', path: '/md-simulation' },
                { name: 'Target Prediction', path: '/tools/prediction' },
                { name: 'Ranking & Leads', path: '/leads' },
            ]
        },
        { name: 'Pricing', path: '/pricing' },
        { name: 'Tools', path: '/tools/converter' },
        { name: 'Blog', path: '/blog' },
        { name: 'Dashboard', path: '/dashboard' },
    ]

    if (isAdmin) {
        navConfig.push({ name: 'Admin', path: '/admin' })
    }

    return (
        <header className={`fixed w-full z-50 transition-all duration-300 ${scrolled ? 'bg-white shadow-md py-3' : 'bg-transparent py-5'}`}>
            <div className="container mx-auto px-4 flex justify-between items-center">
                {/* Logo */}
                <Link to="/" className="flex items-center space-x-3 group">
                    <div className="text-3xl transform group-hover:scale-110 transition-transform duration-300">🧬</div>
                    <h1 className="text-xl font-bold tracking-tight text-slate-900 transition-colors">
                        Bio<span className="text-primary-600">Dockify</span>
                    </h1>
                </Link>

                {/* Desktop Navigation */}
                <nav className="hidden md:flex items-center space-x-5">
                    {navConfig.map((item) => (
                        <div key={item.name} className="relative group">
                            {item.dropdown ? (
                                <button
                                    className="flex items-center gap-1 font-medium text-sm w-max uppercase tracking-wide transition-colors text-slate-900 hover:text-primary-600"
                                >
                                    {item.name}
                                    <ChevronDown size={14} />
                                </button>
                            ) : (
                                <Link
                                    to={item.path}
                                    className={`font-medium text-sm w-max uppercase tracking-wide transition-colors ${(location.pathname === item.path && item.name !== 'Home') ? 'text-primary-600 font-bold' :
                                        'text-slate-900 hover:text-primary-600'
                                        }`}
                                >
                                    {item.name}
                                </Link>
                            )}

                            {/* Dropdown Menu */}
                            {item.dropdown && (
                                <div className="absolute left-0 mt-2 w-48 bg-white rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform translate-y-2 group-hover:translate-y-0 border border-slate-100">
                                    <div className="py-2">
                                        {item.dropdown.map((subItem) => (
                                            <Link
                                                key={subItem.name}
                                                to={subItem.path}
                                                className="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-primary-600"
                                            >
                                                {subItem.name}
                                            </Link>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                </nav>

                {/* Right Side Actions */}
                <div className="flex items-center gap-4">
                    {user ? (
                        <div className="relative group">
                            <button className="flex items-center gap-2 font-bold text-slate-900 hover:text-primary-600 transition-colors">
                                <span>Hi, {user.email?.split('@')[0]}</span>
                                <ChevronDown size={16} />
                            </button>

                            {/* User Dropdown */}
                            <div className="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-slate-100 py-1 overflow-hidden invisible opacity-0 group-hover:visible group-hover:opacity-100 transition-all transform translate-y-2 group-hover:translate-y-0">
                                <div className="px-4 py-2 border-b border-slate-50 bg-slate-50/50">
                                    <p className="text-xs font-bold text-slate-500 uppercase">My Account</p>
                                </div>
                                <Link to="/profile" className="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-primary-600">
                                    👤 Profile
                                </Link>
                                <Link to="/billing" className="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-primary-600">
                                    💳 Billing & Credits
                                </Link>
                                <Link to="/support" className="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-primary-600">
                                    🎫 Support Tickets
                                </Link>
                                <div className="border-t border-slate-50 my-1"></div>
                                <Link to="/dashboard" className="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-primary-600">
                                    📊 Dashboard
                                </Link>
                                <button
                                    onClick={async () => {
                                        await supabase.auth.signOut()
                                        window.location.href = '/'
                                    }}
                                    className="block w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                                >
                                    Sign Out
                                </button>
                            </div>
                        </div>
                    ) : (
                        <Link to="/login" className="font-bold text-slate-900 hover:text-primary-600 transition-colors">Log In</Link>
                    )}
                    <Link to="/dock/new" className="hidden sm:block px-6 py-2.5 bg-primary-600 text-white rounded-xl font-bold text-sm shadow-lg shadow-primary-600/20 hover:bg-primary-700 hover:-translate-y-0.5 transition-all">
                        Start Docking
                    </Link>
                </div>
            </div>
        </header>
    )
}
