import { Link, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'

export default function Header() {
    const [scrolled, setScrolled] = useState(false)
    const location = useLocation()

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20)
        }
        window.addEventListener('scroll', handleScroll)
        return () => window.removeEventListener('scroll', handleScroll)
    }, [])

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
                    {['Features', 'Pricing', 'Tools', 'Blog', 'Dashboard'].map((item) => (
                        <Link
                            key={item}
                            to={
                                item === 'Tools' ? '/tools/converter' :
                                    item === 'Dashboard' ? '/dashboard' :
                                        item === 'Blog' ? '/blog' :
                                            `/#${item.toLowerCase()}`
                            }
                            className={`font-medium text-sm uppercase tracking-wide transition-colors ${scrolled ? 'text-slate-600 hover:text-primary-600' : 'text-slate-700 hover:text-primary-600'}`}
                        >
                            {item}
                        </Link>
                    ))}
                </nav>

                <div className="flex items-center gap-4">
                    <Link to="/login" className={`font-bold transition-colors ${scrolled ? 'text-slate-700 hover:text-primary-600' : 'text-slate-700 hover:text-primary-600'}`}>Log In</Link>
                    <Link to="/dock/new" className="px-6 py-2.5 bg-primary-600 text-white rounded-xl font-bold text-sm shadow-lg shadow-primary-600/20 hover:bg-primary-700 hover:-translate-y-0.5 transition-all">Start Docking</Link>
                </div>
            </div>
        </header>
    )
}
