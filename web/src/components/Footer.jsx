import { Link } from 'react-router-dom'

export default function Footer() {
    return (
        <footer className="bg-gray-900 text-white py-12">
            <div className="container mx-auto px-4">
                <div className="grid md:grid-cols-4 gap-8 mb-8">
                    <div>
                        <div className="flex items-center space-x-2 mb-4">
                            <div className="text-2xl">🧬</div>
                            <h2 className="text-xl font-bold">BioDockify</h2>
                        </div>
                        <p className="text-gray-400 text-sm">
                            Democratizing drug discovery with cloud-native molecular docking tools.
                        </p>
                    </div>
                    <div>
                        <h3 className="font-bold mb-4">Product</h3>
                        <ul className="space-y-2 text-gray-400 text-sm">
                            <li><Link to="/#features" className="hover:text-white">Features</Link></li>
                            <li><Link to="/#pricing" className="hover:text-white">Pricing</Link></li>
                            <li><Link to="/tools/converter" className="hover:text-white">SDF Converter</Link></li>
                            <li><Link to="/admin" className="hover:text-white">Admin</Link></li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="font-bold mb-4">Company</h3>
                        <ul className="space-y-2 text-gray-400 text-sm">
                            <li><Link to="/about" className="hover:text-white">About Us</Link></li>
                            <li><Link to="/blog" className="hover:text-white">Blog</Link></li>
                            <li><Link to="/contact" className="hover:text-white">Contact</Link></li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="font-bold mb-4">Legal</h3>
                        <ul className="space-y-2 text-gray-400 text-sm">
                            <li><Link to="/privacy" className="hover:text-white">Privacy Policy</Link></li>
                            <li><Link to="/terms" className="hover:text-white">Terms of Service</Link></li>
                        </ul>
                    </div>
                </div>
                <div className="border-t border-gray-800 pt-8 text-center text-gray-500 text-sm">
                    © {new Date().getFullYear()} BioDockify. All rights reserved.
                </div>
            </div>
        </footer>
    )
}
