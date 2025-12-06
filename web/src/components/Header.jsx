import { Link, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'

export default function Header() {
    const [scrolled, setScrolled] = useState(false)
    const [isAdmin, setIsAdmin] = useState(false)
    const [user, setUser] = useState(null)
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

    const navItems = ['Features', 'AI Analysis', '3D Viewer', 'Tools', 'Blog', 'Dashboard']
    if (isAdmin) {
        navItems.push('Admin')
    }

    return (
        <header className={`fixed w-full z-50 transition-all duration-300 ${scrolled ? 'bg-white shadow-md py-3' : 'bg-transparent py-5'}`}>
            <div className="container mx-auto px-4 flex justify-between items-center">
                <Link to="/" className="flex items-center space-x-3 group">
                    <div className="text-3xl transform group-hover:scale-110 transition-transform duration-300">🧬</div>
                    <h1 className={`text-2xl font-bold tracking-tight ${scrolled ? 'text-slate-900' : 'text-slate-900'} transition-colors`}>
                        Bio<span className="text-primary-600">Dockify</span>
                    </h1>
                </Link>

                <nav className="hidden md:flex items-center space-x-8">
                    {navItems.map((item) => (
                        <Link
                            key={item}
                            to={
                                item === 'Admin' ? '/admin' :
                                    item === 'Tools' ? '/tools/converter' :
                                        item === 'Dashboard' ? '/dashboard' :
                                            item === 'Blog' ? '/blog' :
                                                item === 'AI Analysis' ? '/ai-analysis' :
                                                    item === '3D Viewer' ? '/3d-viewer' :
                                                        `/#${item.toLowerCase()}`
                            }
                            className={`font-medium text-sm uppercase tracking-wide transition-colors ${(location.pathname.includes(item.toLowerCase()) && item !== 'Features') ? 'text-primary-600 font-bold' :
                                    scrolled ? 'text-slate-600 hover:text-primary-600' : 'text-slate-700 hover:text-primary-600'
                                }`}
                        >
                            {item}
                        </Link>
                    ))}
                </nav>

                <div className="flex items-center gap-4">
                    {user ? (
                        <div className="flex items-center gap-2">
                            <Link to="/profile" className={`font-bold transition-colors ${scrolled ? 'text-slate-700 hover:text-primary-600' : 'text-slate-700 hover:text-primary-600'}`}>
                                Hi, {user.email?.split('@')[0]}
                            </Link>
                        </div>
                    ) : (
                        <Link to="/login" className={`font-bold transition-colors ${scrolled ? 'text-slate-700 hover:text-primary-600' : 'text-slate-700 hover:text-primary-600'}`}>Log In</Link>
                    )}
                    <Link to="/dock/new" className="px-6 py-2.5 bg-primary-600 text-white rounded-xl font-bold text-sm shadow-lg shadow-primary-600/20 hover:bg-primary-700 hover:-translate-y-0.5 transition-all">Start Docking</Link>
                </div>
            </div>
        </header>
    )
}
