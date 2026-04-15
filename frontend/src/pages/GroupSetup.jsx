import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

const STEP_LABELS = ['Group', 'Rate', 'Strategy', 'Results']

const AVATAR_COLORS = [
  '#f59e0b', '#10b981', '#3b82f6', '#ec4899',
  '#8b5cf6', '#f97316', '#06b6d4', '#84cc16',
  '#ef4444', '#a855f7',
]

/**
 * GroupSetup page.
 * Creates a name slot for each group member before rating begins.
 */
export default function GroupSetup({ groupSize, members, setMembers }) {
  const navigate = useNavigate()

  const [names, setNames] = useState(
    Array.from({ length: groupSize }, (_, i) =>
      members[i]?.name || `Person ${i + 1}`
    )
  )

  useEffect(() => {
    setNames((prev) =>
      Array.from({ length: groupSize }, (_, i) => prev[i] || `Person ${i + 1}`)
    )
  }, [groupSize])

  const updateName = (index, value) => {
    setNames((prev) => {
      const updated = [...prev]
      updated[index] = value
      return updated
    })
  }

  const handleStart = () => {
    const initialMembers = names.map((name, i) => ({
      id: `member_${i}`,
      name: name.trim() || `Person ${i + 1}`,
      ratings: {},
    }))
    setMembers(initialMembers)
    navigate('/rate/0')
  }

  return (
    <div className="min-h-screen bg-noir-950 flex flex-col">

      {/* Nav */}
      <nav className="flex items-center justify-between px-8 py-6 border-b border-white/5">
        <button
          onClick={() => navigate('/')}
          className="font-display text-xl text-amber-400 tracking-wider"
          style={{ letterSpacing: '0.15em' }}
        >
          GROUP REC
        </button>
        <div className="flex items-center gap-2">
          {STEP_LABELS.map((step, i) => (
            <React.Fragment key={step}>
              <span
                className={`font-mono text-xs uppercase tracking-widest ${
                  i === 0 ? 'text-amber-400' : 'text-white/20'
                }`}
              >
                {step}
              </span>
              {i < STEP_LABELS.length - 1 && (
                <span className="text-white/10 text-xs">&rsaquo;</span>
              )}
            </React.Fragment>
          ))}
        </div>
      </nav>

      <main className="flex-1 flex flex-col items-center justify-center px-6 py-16">

        <p className="font-mono text-amber-400 text-xs uppercase tracking-widest mb-4 fade-up">
          Step 1 of 4
        </p>

        <h2 className="font-display text-4xl md:text-5xl text-cream mb-4 text-center fade-up fade-up-delay-1">
          Who's watching tonight?
        </h2>

        <p className="font-body text-white/40 mb-12 text-center max-w-sm fade-up fade-up-delay-2">
          Name each person in your group. They will each rate a few movies next.
        </p>

        <div className="w-full max-w-lg space-y-3 fade-up fade-up-delay-3">
          {names.map((name, i) => (
            <div key={i} className="flex items-center gap-4">
              <div
                className="w-10 h-10 rounded-full flex items-center justify-center font-display text-noir-950 font-bold text-sm flex-shrink-0"
                style={{ backgroundColor: AVATAR_COLORS[i] }}
              >
                {name.trim()[0]?.toUpperCase() || (i + 1)}
              </div>
              <input
                type="text"
                value={name}
                onChange={(e) => updateName(i, e.target.value)}
                placeholder={`Person ${i + 1}`}
                maxLength={24}
                className="flex-1 bg-noir-800 border border-white/10 focus:border-amber-400/60 rounded-xl px-4 py-3 font-body text-cream placeholder-white/20 outline-none transition-colors text-base"
              />
            </div>
          ))}
        </div>

        <button
          onClick={handleStart}
          className="mt-12 group bg-amber-500 hover:bg-amber-400 text-noir-950 font-body font-semibold text-lg px-12 py-4 rounded-full transition-all duration-300 hover:scale-105"
          onMouseEnter={(e) => e.currentTarget.style.boxShadow = '0 0 40px rgba(245,158,11,0.4)'}
          onMouseLeave={(e) => e.currentTarget.style.boxShadow = 'none'}
        >
          Let Everyone Rate
          <span className="ml-3 inline-block transition-transform group-hover:translate-x-1">
            &rarr;
          </span>
        </button>
      </main>
    </div>
  )
}
