import { Link } from 'react-router-dom'

export default function AboutPage() {
    return (
        <div className="min-h-screen bg-blue-mesh pt-24 pb-12">
            <div className="container mx-auto px-4">
                <div className="max-w-3xl mx-auto">
                    <h1 className="text-4xl font-extrabold text-white mb-8 text-center tracking-tight">About BioDockify</h1>

                    <div className="glass-modern p-8 mb-8 rounded-2xl">
                        <p className="text-lg text-gray-700 mb-6 leading-relaxed font-light">
                            <strong className="text-cyan-400 font-bold">BioDockify</strong> is a cloud-native molecular docking platform designed to democratize drug discovery for researchers, students, and institutions in India and beyond.
                        </p>
                        <p className="text-lg text-gray-700 mb-6 leading-relaxed font-light">
                            Our mission is to remove the computational barriers to pharmaceutical research by providing high-performance, affordable, and easy-to-use tools powered by AWS cloud infrastructure.
                        </p>
                        <p className="text-lg text-gray-700 leading-relaxed font-light">
                            Whether you are a student learning the basics of CADD or a research scientist screening thousands of compounds, BioDockify scales with your needs.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="glass-modern p-6 rounded-2xl hover:border-cyan-400/50 transition-colors">
                            <div className="text-3xl mb-4 bg-blue-900/50 w-12 h-12 rounded-xl flex items-center justify-center border border-blue-700/50">🚀</div>
                            <h3 className="text-xl font-bold text-white mb-2">Our Vision</h3>
                            <p className="text-gray-700">
                                To accelerate the discovery of life-saving medicines by making advanced computational chemistry tools accessible to everyone.
                            </p>
                        </div>
                        <div className="glass-modern p-6 rounded-2xl hover:border-cyan-400/50 transition-colors">
                            <div className="text-3xl mb-4 bg-blue-900/50 w-12 h-12 rounded-xl flex items-center justify-center border border-blue-700/50">🤝</div>
                            <h3 className="text-xl font-bold text-white mb-2">Our Values</h3>
                            <p className="text-gray-700">
                                Innovation, Accessibility, Transparency, and Scientific Integrity drive everything we do.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
