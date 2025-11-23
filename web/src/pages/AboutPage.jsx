import { Link } from 'react-router-dom'

export default function AboutPage() {
    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
            <div className="container mx-auto px-4 py-12">
                <div className="max-w-3xl mx-auto">
                    <h1 className="text-4xl font-bold text-gray-900 mb-8 text-center">About CloudVina</h1>

                    <div className="bg-white rounded-xl shadow-sm p-8 border border-gray-100 mb-8">
                        <p className="text-lg text-gray-700 mb-6 leading-relaxed">
                            CloudVina is a cloud-native molecular docking platform designed to democratize drug discovery for researchers, students, and institutions in India and beyond.
                        </p>
                        <p className="text-lg text-gray-700 mb-6 leading-relaxed">
                            Our mission is to remove the computational barriers to pharmaceutical research by providing high-performance, affordable, and easy-to-use tools powered by AWS cloud infrastructure.
                        </p>
                        <p className="text-lg text-gray-700 leading-relaxed">
                            Whether you are a student learning the basics of CADD or a research scientist screening thousands of compounds, CloudVina scales with your needs.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
                            <div className="text-3xl mb-4">🚀</div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">Our Vision</h3>
                            <p className="text-gray-600">
                                To accelerate the discovery of life-saving medicines by making advanced computational chemistry tools accessible to everyone.
                            </p>
                        </div>
                        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
                            <div className="text-3xl mb-4">🤝</div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">Our Values</h3>
                            <p className="text-gray-600">
                                Innovation, Accessibility, Transparency, and Scientific Integrity drive everything we do.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
