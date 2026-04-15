import React from 'react'
import { useNavigate } from 'react-router-dom'

/**
 * Landing Page
 * Hero section with group size selector and CTA.
 */
export default function Landing({ groupSize, setGroupSize }) {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'linear-gradient(135deg, #0f2027, #203a43, #2c5364)' }}>
    
      {/* Nav */}
      <nav className="flex items-center justify-between px-8 py-6">
        <span
          className="font-display text-xl text-amber-400 tracking-wider"
          style={{ letterSpacing: '0.15em' }}
        >
          GROUP REC
        </span>
        <span className="font-mono text-xs text-white/30 uppercase tracking-widest">
          Movie Night, Solved
        </span>
      </nav>

      {/* Hero */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 text-center">

        {/* Film Strip Decoration */}
        <div className="flex gap-2 mb-12 opacity-20">
          {[...Array(12)].map((_, i) => (
            <div
              key={i}
              className="w-5 h-8 border border-amber-400/60 rounded-sm"
            />
          ))}
        </div>

        <p className="fade-up fade-up-delay-1 font-mono text-amber-400 text-lg uppercase tracking-widest mb-6">
          The group recommender
        </p>

        <h1
          className="fade-up fade-up-delay-2 font-display text-6xl md:text-8xl text-cream leading-loose mb-6"
          style={{ textShadow: '0 0 60px rgba(245,158,11,0.08)' }}
        >
          Stop arguing
          <br />
          about what
          <br />
          to watch...
        </h1>

        <p className="fade-up fade-up-delay-3 font-body text-white/50 text-lg max-w-md mx-auto mb-16 leading-relaxed">
          Group Rec learns every person's taste and fairly finds the one film
          your whole group will actually enjoy.
        </p>

        {/* Group Size Selector */}
        <div className="fade-up fade-up-delay-4 w-full max-w-md mb-10">
          <div
            className="border border-white/10 rounded-2xl p-8"
            style={{ background: 'rgba(57, 255, 20, 0.3)', boxShadow: 'none' }}
          >
            <p className="font-mono text-xs uppercase tracking-widest text-white/40 mb-4">
              How many people are watching?
            </p>

            <div className="flex items-center justify-between mb-4">
              <span className="font-display text-5xl text-amber-400">
                {groupSize}
              </span>
              <span className="font-body text-white/30 text-sm">people</span>
            </div>

            <input
              type="range"
              min={2}
              max={10}
              value={groupSize}
              onChange={(e) => setGroupSize(Number(e.target.value))}
              className="w-full accent-amber-400 cursor-pointer"
            />

            <div className="flex justify-between mt-2">
              <span className="font-mono text-xs text-white/20">2</span>
              <span className="font-mono text-xs text-white/20">10</span>
            </div>
          </div>
        </div>

        <button
          onClick={() => navigate('/setup')}
          className="fade-up fade-up-delay-4 group relative bg-amber-500 hover:bg-amber-400 text-noir-950 font-body font-semibold text-lg px-12 py-4 rounded-full transition-all duration-300 hover:scale-105 mb-16"
          style={{ boxShadow: '0 0 0 rgba(245,158,11,0)' }}
          onMouseEnter={(e) => e.currentTarget.style.boxShadow = '0 0 40px rgba(245,158,11,0.4)'}
          onMouseLeave={(e) => e.currentTarget.style.boxShadow = '0 0 0 rgba(245,158,11,0)'}
        >
          Build Your Group
          <span className="ml-3 inline-block transition-transform group-hover:translate-x-1">
            &rarr;
          </span>
        </button>

      </main>

      {/* Bottom Rule */}
      <div className="h-px bg-gradient-to-r from-transparent via-amber-400/30 to-transparent" />
    </div>
  )
}
