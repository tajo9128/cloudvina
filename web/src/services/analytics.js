// Analytics Service - DISABLED by User Request
// This file now exports dummy functions to prevent import errors

export const initAnalytics = () => {
    console.log("Analytics disabled")
}

export const trackEvent = (eventName, props = {}) => {
    // console.log("Analytics event skipped:", eventName, props)
}

export const identifyUser = (userId, traits = {}) => {
    // console.log("Analytics identify skipped:", userId)
}

export default {
    init: initAnalytics,
    track: trackEvent,
    identify: identifyUser
}
