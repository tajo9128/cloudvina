import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'
import { Link } from 'react-router-dom'

export default function SupportPage() {
    const [tickets, setTickets] = useState([])
    const [loading, setLoading] = useState(true)
    const [creating, setCreating] = useState(false)
    const [showForm, setShowForm] = useState(false)
    const [newTicket, setNewTicket] = useState({ subject: '', category: 'General', message: '' })

    useEffect(() => {
        fetchTickets()
    }, [])

    const fetchTickets = async () => {
        try {
            const { data: { user } } = await supabase.auth.getUser()
            if (!user) return

            const { data, error } = await supabase
                .from('support_tickets')
                .select('*')
                .eq('user_id', user.id)
                .order('created_at', { ascending: false })

            if (error) throw error
            setTickets(data || [])
        } catch (error) {
            console.error("Error fetching tickets:", error)
        } finally {
            setLoading(false)
        }
    }

    const handleCreateTicket = async (e) => {
        e.preventDefault()
        setCreating(true)
        try {
            const { data: { user } } = await supabase.auth.getUser()

            const { error } = await supabase
                .from('support_tickets')
                .insert({
                    user_id: user.id,
                    subject: newTicket.subject,
                    category: newTicket.category,
                    message: newTicket.message,
                    status: 'open'
                })

            if (error) throw error

            setShowForm(false)
            setNewTicket({ subject: '', category: 'General', message: '' })
            fetchTickets() // Refresh list

        } catch (error) {
            alert('Failed to create ticket: ' + error.message)
        } finally {
            setCreating(false)
        }
    }

    const getStatusBadge = (status) => {
        const styles = {
            open: 'bg-blue-100 text-blue-800',
            resolved: 'bg-green-100 text-green-800',
            closed: 'bg-slate-100 text-slate-800',
            in_progress: 'bg-yellow-100 text-yellow-800'
        }
        return (
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold uppercase ${styles[status] || styles.open}`}>
                {status.replace('_', ' ')}
            </span>
        )
    }

    if (loading) return <div className="min-h-screen bg-slate-50 flex items-center justify-center">Loading...</div>

    return (
        <div className="min-h-screen bg-slate-50 py-12">
            <div className="container mx-auto px-4">
                <div className="max-w-6xl mx-auto">
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
                        <div>
                            <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Support Center</h1>
                            <p className="text-slate-500 mt-1">Track issues and get help from our team.</p>
                        </div>
                        <button
                            onClick={() => setShowForm(!showForm)}
                            className="btn-primary px-6 py-2.5 rounded-xl font-bold shadow-lg shadow-primary-600/20 flex items-center gap-2"
                        >
                            <span>{showForm ? 'Cancel' : '➕ Create New Ticket'}</span>
                        </button>
                    </div>

                    <div className="grid lg:grid-cols-3 gap-8">
                        {/* Ticket List (Left 2/3) */}
                        <div className="lg:col-span-2">
                            {tickets.length === 0 ? (
                                <div className="bg-white p-12 rounded-2xl shadow-sm border border-slate-200 text-center">
                                    <div className="text-5xl mb-4">🎫</div>
                                    <h3 className="text-xl font-bold text-slate-900 mb-2">No tickets yet</h3>
                                    <p className="text-slate-500">Need help? Create a ticket to get started.</p>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    {tickets.map(ticket => (
                                        <div key={ticket.id} className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 hover:shadow-md transition">
                                            <div className="flex justify-between items-start mb-2">
                                                <div className="flex items-center gap-3">
                                                    {getStatusBadge(ticket.status)}
                                                    <span className="text-xs text-slate-400 font-medium">#{ticket.id.slice(0, 8)}</span>
                                                </div>
                                                <span className="text-xs text-slate-400">
                                                    {new Date(ticket.created_at).toLocaleDateString()}
                                                </span>
                                            </div>
                                            <h3 className="text-lg font-bold text-slate-900 mb-2">{ticket.subject}</h3>
                                            <p className="text-slate-600 text-sm line-clamp-2">{ticket.message}</p>
                                            <div className="mt-4 pt-4 border-t border-slate-50 flex justify-between items-center text-xs">
                                                <span className="text-slate-500 font-medium bg-slate-100 px-2 py-1 rounded">
                                                    Category: {ticket.category}
                                                </span>
                                                {/* Future: Link to detail view */}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Create Form / Info (Right 1/3) */}
                        <div className="space-y-6">
                            {showForm ? (
                                <div className="bg-white p-6 rounded-2xl shadow-xl border border-slate-200 animate-fade-in-up">
                                    <h3 className="text-lg font-bold text-slate-900 mb-4">New Support Ticket</h3>
                                    <form onSubmit={handleCreateTicket} className="space-y-4">
                                        <div>
                                            <label className="block text-xs font-bold text-slate-700 mb-1">Subject</label>
                                            <input
                                                type="text"
                                                required
                                                value={newTicket.subject}
                                                onChange={e => setNewTicket({ ...newTicket, subject: e.target.value })}
                                                className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:ring-2 focus:ring-primary-500 outline-none"
                                                placeholder="Brief summary of issue"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs font-bold text-slate-700 mb-1">Category</label>
                                            <select
                                                value={newTicket.category}
                                                onChange={e => setNewTicket({ ...newTicket, category: e.target.value })}
                                                className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:ring-2 focus:ring-primary-500 outline-none"
                                            >
                                                <option value="General">General Inquiry</option>
                                                <option value="Billing">Billing & Credits</option>
                                                <option value="Technical">Technical Issue</option>
                                                <option value="Feature Request">Feature Request</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-xs font-bold text-slate-700 mb-1">Message</label>
                                            <textarea
                                                required
                                                rows="5"
                                                value={newTicket.message}
                                                onChange={e => setNewTicket({ ...newTicket, message: e.target.value })}
                                                className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:ring-2 focus:ring-primary-500 outline-none resize-none"
                                                placeholder="Describe your issue in detail..."
                                            />
                                        </div>
                                        <button
                                            type="submit"
                                            disabled={creating}
                                            className="w-full btn-primary py-2 rounded-xl font-bold shadow-lg"
                                        >
                                            {creating ? 'Submitting...' : 'Submit Ticket'}
                                        </button>
                                    </form>
                                </div>
                            ) : (
                                <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-6 text-white shadow-lg">
                                    <h3 className="font-bold text-lg mb-2">Need immediate help?</h3>
                                    <p className="text-slate-300 text-sm mb-4">
                                        Check our comprehensive documentation or email us directly for urgent matters.
                                    </p>
                                    <Link to="/contact" className="text-primary-400 font-bold text-sm hover:text-primary-300">
                                        Contact Page →
                                    </Link>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
