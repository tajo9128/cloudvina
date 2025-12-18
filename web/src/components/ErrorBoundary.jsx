import React from 'react'
import { Link } from 'react-router-dom'

export default class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props)
        this.state = { hasError: false, error: null }
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error }
    }

    componentDidCatch(error, errorInfo) {
        console.error("Uncaught error:", error, errorInfo)
    }

    handleReset = () => {
        this.setState({ hasError: false, error: null })
        // Optional: Force reload or clear storage if needed
        window.location.reload()
    }

    handleLogout = () => {
        // Force logout if the error is auth/state related
        localStorage.clear()
        window.location.href = '/'
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-lg w-full p-8 text-center border border-slate-200">
                        <div className="w-20 h-20 bg-red-50 rounded-full flex items-center justify-center mx-auto mb-6 text-4xl">
                            💣
                        </div>
                        <h1 className="text-2xl font-bold text-slate-900 mb-2">Something went wrong</h1>
                        <p className="text-slate-500 mb-6">
                            We're sorry, but the application encountered an unexpected error.
                            <br />
                            <span className="text-xs font-mono bg-slate-100 px-2 py-1 rounded text-red-500 mt-2 inline-block max-w-full truncate">
                                {this.state.error?.message || "Unknown Error"}
                            </span>
                        </p>

                        <div className="flex gap-4 justify-center">
                            <button
                                onClick={this.handleReset}
                                className="px-6 py-2 bg-primary-600 text-white font-bold rounded-xl hover:bg-primary-700 transition"
                            >
                                Reload Page
                            </button>
                            <button
                                onClick={this.handleLogout}
                                className="px-6 py-2 bg-white border border-slate-200 text-slate-700 font-bold rounded-xl hover:bg-slate-50 transition"
                            >
                                Clear Session
                            </button>
                        </div>
                    </div>
                </div>
            )
        }

        return this.props.children
    }
}
