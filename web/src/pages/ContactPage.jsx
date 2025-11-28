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
        <div className="min-h-screen bg-blue-mesh pt-24 pb-12">
            <div className="container mx-auto px-4">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-12">
                        <h1 className="text-4xl font-bold text-deep-navy-900 mb-4">Contact Us</h1>
                        <p className="text-slate-600 max-w-2xl mx-auto text-lg">
                            Have questions about CloudVina? Need help with your docking jobs?
                            We're here to help.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-12">
                        {/* Contact Info */}
                        <div className="space-y-8">
                            <div className="glass-card p-8">
                                <h3 className="text-xl font-bold text-deep-navy-900 mb-6">Get in Touch</h3>
                                <div className="space-y-6">
                                    <div className="flex items-start">
                                        <div className="text-2xl mr-4 bg-blue-50 w-10 h-10 rounded-lg flex items-center justify-center">📧</div>
                                        <div>
                                            <p className="font-bold text-deep-navy-900">Email</p>
                                            <a href="mailto:cloudvina2025@gmail.com" className="text-blue-600 hover:text-blue-700 font-medium">
                                                cloudvina2025@gmail.com
                                            </a>
                                            <p className="text-sm text-slate-500 mt-1">
                                                We usually respond within 24 hours.
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex items-start">
                                        <div className="text-2xl mr-4 bg-blue-50 w-10 h-10 rounded-lg flex items-center justify-center">📍</div>
                                        <div>
                                            <p className="font-bold text-deep-navy-900">Location</p>
                                            <p className="text-slate-600">
                                                Hyderabad, India<br />
                                                (Cloud-based operations)
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl shadow-lg shadow-blue-200 p-8 text-white relative overflow-hidden">
                                <div className="absolute top-0 right-0 w-32 h-32 bg-white opacity-10 rounded-full -mr-16 -mt-16 blur-2xl"></div>
                                <h3 className="text-xl font-bold mb-4 relative z-10">Support for Students</h3>
                                <p className="mb-4 opacity-90 relative z-10">
                                    Are you a student or researcher from an Indian university?
                                    We offer special support and extended credits for academic projects.
                                </p>
                                <p className="font-bold relative z-10 bg-white/20 inline-block px-3 py-1 rounded-lg">
                                    Mention your institution!
                                </p>
                            </div>
                        </div>

                        {/* Contact Form */}
                        <div className="glass-card p-8">
                            <form onSubmit={handleSubmit} className="space-y-6">
                                <div>
                                    <label className="block text-sm font-bold text-deep-navy-900 mb-2">
                                        Your Name
                                    </label>
                                    <input
                                        type="text"
                                        required
                                        className="w-full px-4 py-3 bg-white/50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition outline-none"
                                        placeholder="Dr. Rajesh Kumar"
                                        value={formData.name}
                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-deep-navy-900 mb-2">
                                        Email Address
                                    </label>
                                    <input
                                        type="email"
                                        required
                                        className="w-full px-4 py-3 bg-white/50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition outline-none"
                                        placeholder="rajesh@iit.ac.in"
                                        value={formData.email}
                                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-deep-navy-900 mb-2">
                                        Subject
                                    </label>
                                    <select
                                        className="w-full px-4 py-3 bg-white/50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition outline-none"
                                        value={formData.subject}
                                        onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                                    >
                                        <option value="" disabled>Select a topic</option>
                                        <option value="General Inquiry">General Inquiry</option>
                                        <option value="Technical Support">Technical Support</option>
                                        <option value="Billing/Credits">Billing & Credits</option>
                                        <option value="Academic Partnership">Academic Partnership</option>
                                        <option value="Bug Report">Bug Report</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-deep-navy-900 mb-2">
                                        Message
                                    </label>
                                    <textarea
                                        required
                                        rows="4"
                                        className="w-full px-4 py-3 bg-white/50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition outline-none"
                                        placeholder="How can we help you?"
                                        value={formData.message}
                                        onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                                    ></textarea>
                                </div>
                                <button
                                    type="submit"
                                    className="w-full btn-blue-glow py-3 rounded-xl font-bold text-lg shadow-lg shadow-blue-200/50"
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
