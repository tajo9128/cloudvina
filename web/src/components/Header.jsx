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
        <header className={`fixed w-full z-50 transition-all duration-300 ${scrolled ? 'glass-dark border-b border-white/10 py-3' : 'bg-transparent py-5'}`}>
            <div className="container mx-auto px-4 flex justify-between items-center">
                <Link to="/" className="flex items-center space-x-3 group">
                    <div className="text-3xl transform group-hover:scale-110 transition-transform duration-300 drop-shadow-[0_0_8px_rgba(0,217,255,0.5)]">🧬</div>
                    <h1 className="text-2xl font-extrabold text-white tracking-tight">Bio<span className="text-cyan-400">Dockify</span></h1>
                </Link>

                <nav className="hidden md:flex items-center space-x-8">
                    <Link to="/#features" className="text-blue-100 hover:text-cyan-400 font-medium transition-colors text-sm uppercase tracking-wide">Features</Link>
                    <Link to="/#pricing" className="text-blue-100 hover:text-cyan-400 font-medium transition-colors text-sm uppercase tracking-wide">Pricing</Link>
                    <Link to="/tools/converter" className="text-blue-100 hover:text-cyan-400 font-medium transition-colors text-sm uppercase tracking-wide">Tools</Link>
                    <Link to="/blog" className="text-blue-100 hover:text-cyan-400 font-medium transition-colors text-sm uppercase tracking-wide">Blog</Link>
                    <Link to="/dashboard" className="text-blue-100 hover:text-cyan-400 font-medium transition-colors text-sm uppercase tracking-wide">Dashboard</Link>
                </nav>

                <div className="flex items-center gap-4">
                    <Link to="/login" className="px-5 py-2 text-white font-bold hover:text-cyan-300 transition-colors">Log In</Link>
                    <Link to="/dock/new" className="btn-cyan px-6 py-2.5 rounded-xl font-bold text-sm shadow-lg shadow-cyan-500/20">Start Docking</Link>
                </div>
            </div>
        </header>
    )
}
