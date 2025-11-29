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
        const subject = encodeURIComponent(`[BioDockify Inquiry] ${formData.subject}`)
        const body = encodeURIComponent(`Name: ${formData.name}\nEmail: ${formData.email}\n\nMessage:\n${formData.message}`)
        window.location.href = `mailto:biodockify@hotmail.com?subject=${subject}&body=${body}`
    }

    return (
        <div className="min-h-screen bg-blue-mesh pt-24 pb-12">
            <div className="container mx-auto px-4">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-12">
                        <h1 className="text-4xl font-extrabold text-white mb-4 tracking-tight">Contact Us</h1>
                        <p className="text-blue-200 max-w-2xl mx-auto text-lg font-light">
                            Have questions about BioDockify? Need help with your docking jobs?
                            We're here to help.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-12">
                        {/* Contact Info */}
                        <div className="space-y-8">
                            <div className="glass-modern p-8 rounded-2xl">
                                <h3 className="text-xl font-bold text-white mb-6">Get in Touch</h3>
                                <div className="space-y-6">
                                    <div className="flex items-start">
                                        <div className="text-2xl mr-4 bg-blue-900/50 w-10 h-10 rounded-lg flex items-center justify-center border border-blue-700/50">📧</div>
                                        <div>
                                            <p className="font-bold text-white">Email</p>
                                            <a href="mailto:biodockify@hotmail.com" className="text-cyan-400 hover:text-cyan-300 font-medium transition-colors">
                                                biodockify@hotmail.com
                                            </a>
                                            <p className="text-sm text-blue-200/60 mt-1">
                                                We usually respond within 24 hours.
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex items-start">
                                        <div className="text-2xl mr-4 bg-blue-900/50 w-10 h-10 rounded-lg flex items-center justify-center border border-blue-700/50">📍</div>
                                        <div>
                                            <p className="font-bold text-white">Location</p>
                                            <p className="text-blue-200/80">
                                                Hyderabad, India<br />
                                                (Cloud-based operations)
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl shadow-lg shadow-cyan-500/20 p-8 text-white relative overflow-hidden">
                                <div className="absolute top-0 right-0 w-32 h-32 bg-white opacity-10 rounded-full -mr-16 -mt-16 blur-2xl"></div>
                                <h3 className="text-xl font-bold mb-4 relative z-10">Support for Students</h3>
                                <p className="mb-4 opacity-90 relative z-10 text-blue-50">
                                    Are you a student or researcher from an Indian university?
                                    We offer special support and extended credits for academic projects.
                                </p>
                                <p className="font-bold relative z-10 bg-white/20 inline-block px-3 py-1 rounded-lg backdrop-blur-sm">
                                    Mention your institution!
                                </p>
                            </div>
                        </div>

                        {/* Contact Form */}
                        <div className="glass-modern p-8 rounded-2xl">
                            <form onSubmit={handleSubmit} className="space-y-6">
                                <div>
                                    <label className="block text-sm font-bold text-white mb-2">
                                        Your Name
                                    </label>
                                    <input
                                        type="text"
                                        required
                                        className="w-full px-4 py-3 bg-blue-900/30 border border-blue-700/50 rounded-xl focus:ring-2 focus:ring-cyan-400 focus:border-cyan-400 transition outline-none text-white placeholder-blue-300/30"
                                        placeholder="Dr. Rajesh Kumar"
                                        value={formData.name}
                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-white mb-2">
                                        Email Address
                                    </label>
                                    <input
                                        type="email"
                                        required
                                        className="w-full px-4 py-3 bg-blue-900/30 border border-blue-700/50 rounded-xl focus:ring-2 focus:ring-cyan-400 focus:border-cyan-400 transition outline-none text-white placeholder-blue-300/30"
                                        placeholder="rajesh@iit.ac.in"
                                        value={formData.email}
                                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-white mb-2">
                                        Subject
                                    </label>
                                    <select
                                        className="w-full px-4 py-3 bg-blue-900/30 border border-blue-700/50 rounded-xl focus:ring-2 focus:ring-cyan-400 focus:border-cyan-400 transition outline-none text-white"
                                        value={formData.subject}
                                        onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                                    >
                                        <option value="" disabled className="bg-deep-navy-900">Select a topic</option>
                                        <option value="General Inquiry" className="bg-deep-navy-900">General Inquiry</option>
                                        <option value="Technical Support" className="bg-deep-navy-900">Technical Support</option>
                                        <option value="Billing/Credits" className="bg-deep-navy-900">Billing & Credits</option>
                                        <option value="Academic Partnership" className="bg-deep-navy-900">Academic Partnership</option>
                                        <option value="Bug Report" className="bg-deep-navy-900">Bug Report</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-white mb-2">
                                        Message
                                    </label>
                                    <textarea
                                        required
                                        rows="4"
                                        className="w-full px-4 py-3 bg-blue-900/30 border border-blue-700/50 rounded-xl focus:ring-2 focus:ring-cyan-400 focus:border-cyan-400 transition outline-none text-white placeholder-blue-300/30"
                                        placeholder="How can we help you?"
                                        value={formData.message}
                                        onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                                    ></textarea>
                                </div>
                                <button
                                    type="submit"
                                    className="w-full btn-cyan py-3 rounded-xl font-bold text-lg shadow-lg shadow-cyan-500/20"
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
