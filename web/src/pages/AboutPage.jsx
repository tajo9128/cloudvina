import { Link } from 'react-router-dom'

export default function AboutPage() {
    return (
        <div className="min-h-screen bg-blue-mesh pt-24 pb-12">
            <div className="container mx-auto px-4">
                <div className="max-w-3xl mx-auto">
                    <h1 className="text-4xl font-bold text-deep-navy-900 mb-8 text-center">About CloudVina</h1>

                    <div className="glass-card p-8 mb-8">
                        <p className="text-lg text-slate-600 mb-6 leading-relaxed">
                            CloudVina is a cloud-native molecular docking platform designed to democratize drug discovery for researchers, students, and institutions in India and beyond.
                        </p>
                        <p className="text-lg text-slate-600 mb-6 leading-relaxed">
                            Our mission is to remove the computational barriers to pharmaceutical research by providing high-performance, affordable, and easy-to-use tools powered by AWS cloud infrastructure.
                        </p>
                        <p className="text-lg text-slate-600 leading-relaxed">
                            Whether you are a student learning the basics of CADD or a research scientist screening thousands of compounds, CloudVina scales with your needs.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="glass-card p-6">
                            <div className="text-3xl mb-4 bg-blue-50 w-12 h-12 rounded-xl flex items-center justify-center">🚀</div>
                            <h3 className="text-xl font-bold text-deep-navy-900 mb-2">Our Vision</h3>
                            <p className="text-slate-600">
                                To accelerate the discovery of life-saving medicines by making advanced computational chemistry tools accessible to everyone.
                            </p>
                        </div>
                        <div className="glass-card p-6">
                            <div className="text-3xl mb-4 bg-blue-50 w-12 h-12 rounded-xl flex items-center justify-center">🤝</div>
                            <h3 className="text-xl font-bold text-deep-navy-900 mb-2">Our Values</h3>
                            <p className="text-slate-600">
                                Innovation, Accessibility, Transparency, and Scientific Integrity drive everything we do.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
