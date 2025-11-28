import { Link } from 'react-router-dom'

export default function Header() {
    return (
        <header className="bg-white shadow-sm fixed w-full z-50">
            <div className="container mx-auto px-4 py-4 flex justify-between items-center">
                <Link to="/" className="flex items-center space-x-2 text-gray-800 hover:text-purple-600 transition">
                    <div className="text-2xl">🧬</div>
                    <h1 className="text-xl font-bold">BioDockify</h1>
                </Link>
                <nav className="hidden md:flex space-x-8">
                    <Link to="/#features" className="text-gray-600 hover:text-purple-600 font-medium">Features</Link>
                    <Link to="/#pricing" className="text-gray-600 hover:text-purple-600 font-medium">Pricing</Link>
                    <Link to="/tools/converter" className="text-gray-600 hover:text-purple-600 font-medium">Tools</Link>
                    <Link to="/dashboard" className="text-gray-600 hover:text-purple-600 font-medium">Dashboard</Link>
                </nav>
                <div className="flex gap-4">
                    <Link to="/login" className="px-4 py-2 text-purple-600 font-medium hover:bg-purple-50 rounded-lg transition">Log In</Link>
                    <Link to="/dock/new" className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition shadow-md">Start Docking</Link>
                </div>
            </div>
        </header>
    )
}
