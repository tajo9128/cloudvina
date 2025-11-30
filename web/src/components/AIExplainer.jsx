import { useState } from 'react'
import { API_URL } from '../config'
import { supabase } from '../supabaseClient'

export default function AIExplainer({ jobId }) {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [showChat, setShowChat] = useState(false)

    const getExplanation = async (question = null) => {
        setLoading(true)

        const { data: { session } } = await supabase.auth.getSession()
        if (!session) return

        try {
            const response = await fetch(`${API_URL}/jobs/${jobId}/explain`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question })
            })

            const reader = response.body.getReader()
            const decoder = new TextDecoder()
            let aiMessage = ''

            // Add user message if question
            if (question) {
                setMessages(prev => [...prev, {
                    role: 'user',
                    content: question
                }])
            }

            // Add initial empty assistant message
            setMessages(prev => [...prev, { role: 'assistant', content: '' }])

            // Stream AI response
            while (true) {
                const { value, done } = await reader.read()
                if (done) break

                const chunk = decoder.decode(value)
                // Parse SSE format
                const lines = chunk.split('\n')
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const content = line.slice(6)
                        aiMessage += content

                        // Update last message
                        setMessages(prev => {
                            const newMsgs = [...prev]
                            newMsgs[newMsgs.length - 1].content = aiMessage
                            return newMsgs
                        })
                    }
                }
            }
        } catch (error) {
            console.error('AI explanation error:', error)
            setMessages(prev => {
                const newMsgs = [...prev]
                // If last message was empty assistant message, update it
                if (newMsgs.length > 0 && newMsgs[newMsgs.length - 1].role === 'assistant') {
                    newMsgs[newMsgs.length - 1].content = '❌ Sorry, I encountered an error. Please try again.'
                } else {
                    newMsgs.push({
                        role: 'assistant',
                        content: '❌ Sorry, I encountered an error. Please try again.'
                    })
                }
                return newMsgs
            })
        } finally {
            setLoading(false)
        }
    }

    const handleAskQuestion = (e) => {
        e.preventDefault()
        if (!input.trim()) return
        getExplanation(input)
        setInput('')
    }

    return (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mt-8">
            {/* Header */}
            <div className="p-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center text-2xl">
                            🤖
                        </div>
                        <div>
                            <h3 className="font-bold text-lg">AI Results Explainer</h3>
                            <p className="text-sm opacity-90">Powered by Grok AI</p>
                        </div>
                    </div>
                    {!showChat && (
                        <button
                            onClick={() => {
                                setShowChat(true)
                                if (messages.length === 0) {
                                    getExplanation()  // Initial explanation
                                }
                            }}
                            className="px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg font-medium transition-colors"
                        >
                            Explain My Results 💬
                        </button>
                    )}
                </div>
            </div>

            {/* Chat Area */}
            {showChat && (
                <div className="flex flex-col h-[500px]">
                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4">
                        {messages.map((msg, idx) => (
                            <div
                                key={idx}
                                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div className={`max-w-[80%] rounded-lg p-3 whitespace-pre-wrap ${msg.role === 'user'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-slate-100 text-slate-900'
                                    }`}>
                                    {msg.content}
                                </div>
                            </div>
                        ))}
                        {loading && (
                            <div className="flex justify-start">
                                <div className="bg-slate-100 rounded-lg p-3">
                                    <div className="flex gap-1">
                                        <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                                        <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-100"></div>
                                        <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-200"></div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Input */}
                    <form onSubmit={handleAskQuestion} className="p-4 border-t border-slate-200 bg-slate-50">
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask a question about your results..."
                                className="flex-1 px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                                disabled={loading}
                            />
                            <button
                                type="submit"
                                disabled={loading || !input.trim()}
                                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                            >
                                Send
                            </button>
                        </div>
                        <div className="mt-2 text-xs text-slate-500 flex gap-2">
                            <span>💡 Try:</span>
                            <button type="button" onClick={() => setInput("What does my binding affinity mean?")} className="hover:text-blue-600 hover:underline">"What does my binding affinity mean?"</button>
                            <button type="button" onClick={() => setInput("Is this a good result?")} className="hover:text-blue-600 hover:underline">"Is this a good result?"</button>
                        </div>
                    </form>
                </div>
            )}
        </div>
    )
}
