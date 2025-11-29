import { Link } from 'react-router-dom'

export default function AboutPage() {
    return (
        <div className="min-h-screen bg-slate-50 pt-32 pb-20">
            <div className="container mx-auto px-4">
                <div className="max-w-3xl mx-auto text-center mb-16">
                    <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6 tracking-tight">About BioDockify</h1>
                    <p className="text-xl text-slate-600 leading-relaxed">
                        Democratizing drug discovery with cloud-native molecular docking.
                    </p>
                </div>

                <div className="max-w-4xl mx-auto">
                    <div className="bg-white p-8 md:p-12 rounded-2xl shadow-sm border border-slate-200 mb-12">
                        <p className="text-lg text-slate-700 mb-6 leading-relaxed">
                            <strong className="text-primary-600 font-bold">BioDockify</strong> is a cloud-native molecular docking platform designed to empower researchers, students, and institutions worldwide.
                        </p>
                        <p className="text-lg text-slate-700 mb-6 leading-relaxed">
                            Our mission is to remove the computational barriers to pharmaceutical research by providing high-performance, affordable, and easy-to-use tools powered by AWS cloud infrastructure.
                        </p>
                        <p className="text-lg text-slate-700 leading-relaxed">
                            Whether you are a student learning the basics of CADD or a research scientist screening thousands of compounds, BioDockify scales with your needs.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-8">
                        <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 hover:border-primary-200 transition-all group">
                            <div className="w-12 h-12 rounded-xl bg-primary-50 text-primary-600 flex items-center justify-center text-2xl mb-6 group-hover:scale-110 transition-transform">
                                🚀
                            </div>
                            <h3 className="text-xl font-bold text-slate-900 mb-3">Our Vision</h3>
                            <p className="text-slate-600 leading-relaxed">
                                To accelerate the discovery of life-saving medicines by making advanced computational chemistry tools accessible to everyone.
                            </p>
                        </div>
                        <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 hover:border-primary-200 transition-all group">
                            <div className="w-12 h-12 rounded-xl bg-secondary-50 text-secondary-600 flex items-center justify-center text-2xl mb-6 group-hover:scale-110 transition-transform">
                                🤝
                            </div>
                            <h3 className="text-xl font-bold text-slate-900 mb-3">Our Values</h3>
                            <p className="text-slate-600 leading-relaxed">
                                Innovation, Accessibility, Transparency, and Scientific Integrity drive everything we do.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
