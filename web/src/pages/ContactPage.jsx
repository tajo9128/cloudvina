import { useState } from 'react'

export default function ContactPage() {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        subject: '',
        message: ''
    })

    const handleSubmit = (e) => {
        e.preventDefault()
        // For MVP, we'll just open the mail client with the filled details
        const subject = encodeURIComponent(`[CloudVina Inquiry] ${formData.subject}`)
        const body = encodeURIComponent(`Name: ${formData.name}\nEmail: ${formData.email}\n\nMessage:\n${formData.message}`)
        window.location.href = `mailto:cloudvina2025@gmail.com?subject=${subject}&body=${body}`
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
            <div className="container mx-auto px-4 py-12">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-12">
                        <h1 className="text-4xl font-bold text-gray-900 mb-4">Contact Us</h1>
                        <p className="text-gray-600 max-w-2xl mx-auto">
                            Have questions about CloudVina? Need help with your docking jobs?
                            We're here to help.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-12">
                        {/* Contact Info */}
                        <div className="space-y-8">
                            <div className="bg-white rounded-xl shadow-sm p-8 border border-gray-100">
                                <h3 className="text-xl font-bold text-gray-900 mb-4">Get in Touch</h3>
                                <div className="space-y-6">
                                    <div className="flex items-start">
                                        <div className="text-2xl mr-4">📧</div>
                                        <div>
                                            <p className="font-medium text-gray-900">Email</p>
                                            <a href="mailto:cloudvina2025@gmail.com" className="text-purple-600 hover:text-purple-700">
                                                cloudvina2025@gmail.com
                                            </a>
                                            <p className="text-sm text-gray-500 mt-1">
                                                We usually respond within 24 hours.
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex items-start">
                                        <div className="text-2xl mr-4">📍</div>
                                        <div>
                                            <p className="font-medium text-gray-900">Location</p>
                                            <p className="text-gray-600">
                                                Hyderabad, India<br />
                                                (Cloud-based operations)
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-purple-600 rounded-xl shadow-lg p-8 text-white">
                                <h3 className="text-xl font-bold mb-4">Support for Students</h3>
                                <p className="mb-4 opacity-90">
                                    Are you a student or researcher from an Indian university?
                                    We offer special support and extended credits for academic projects.
                                </p>
                                <p className="font-medium">
                                    Mention your institution when contacting us!
                                </p>
                            </div>
                        </div>

                        {/* Contact Form */}
                        <div className="bg-white rounded-xl shadow-lg p-8">
                            <form onSubmit={handleSubmit} className="space-y-6">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Your Name
                                    </label>
                                    <input
                                        type="text"
                                        required
                                        className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition"
                                        placeholder="Dr. Rajesh Kumar"
                                        value={formData.name}
                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Email Address
                                    </label>
                                    <input
                                        type="email"
                                        required
                                        className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition"
                                        placeholder="rajesh@iit.ac.in"
                                        value={formData.email}
                                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Subject
                                    </label>
                                    <select
                                        className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition"
                                        value={formData.subject}
                                        onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                                    >
                                        <option value="General Inquiry">General Inquiry</option>
                                        <option value="Technical Support">Technical Support</option>
                                        <option value="Billing/Credits">Billing & Credits</option>
                                        <option value="Academic Partnership">Academic Partnership</option>
                                        <option value="Bug Report">Bug Report</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Message
                                    </label>
                                    <textarea
                                        required
                                        rows="4"
                                        className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition"
                                        placeholder="How can we help you?"
                                        value={formData.message}
                                        onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                                    ></textarea>
                                </div>
                                <button
                                    type="submit"
                                    className="w-full bg-purple-600 text-white py-3 rounded-lg font-bold hover:bg-purple-700 transition shadow-md"
                                >
                                    Send Message
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
