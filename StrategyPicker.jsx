import React from 'react'
import { useNavigate } from 'react-router-dom'

const STEP_LABELS = ['Group', 'Rate', 'Strategy', 'Results']

const STRATEGIES = [
  {
    id: 'least_misery',
    label: 'No One Hates It',
    subtitle: 'Least Misery',
    icon: 'LM',
    description:
      'The group score is the lowest individual score. Guarantees nobody walks away miserable - but can be conservative when tastes vary widely.',
    bestFor: 'Groups with one very picky person',
    color: '#3b82f6',
  },
  {
    id: 'average',
    label: 'Most Overall Happy',
    subtitle: 'Average Satisfaction',
    icon: 'AS',
    description:
      'The group score is the average of all individual scores. Maximizes total happiness across the group - but can steamroll one person with minority taste.',
    bestFor: 'Groups with similar tastes',
    color: '#10b981',
  },
  {
    id: 'fairness_aware',
    label: 'The Balanced Pick',
    subtitle: 'Fairness-Aware',
    icon: 'FA',
    description:
      'Blends average satisfaction with least misery. You control the balance - slide toward fairness or toward maximum happiness.',
    bestFor: 'Most groups (recommended)',
    color: '#f59e0b',
    hasSlider: true,
  },
]

/**
 * StrategyPicker page.
 * Three strategy cards with plain-English explanations.
 * Fairness-Aware card exposes an alpha slider when selected.
 */
export default function StrategyPicker({ strategy, setStrategy, alpha, setAlpha }) {
  const navigate = useNavigate()

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
              <span className={`font-mono text-xs uppercase tracking-widest ${i === 2 ? 'text-amber-400' : 'text-white/20'}`}>
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
          Step 3 of 4
        </p>

        <h2 className="font-display text-4xl md:text-5xl text-cream mb-4 text-center fade-up fade-up-delay-1">
          How should we decide?
        </h2>

        <p className="font-body text-white/40 mb-12 text-center max-w-md fade-up fade-up-delay-2">
          Each strategy makes a different tradeoff between total group happiness
          and protecting the most dissatisfied person.
        </p>

        {/* Strategy cards */}
        <div className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-3 gap-4 mb-10 fade-up fade-up-delay-3">
          {STRATEGIES.map((s) => {
            const selected = strategy === s.id
            return (
              <button
                key={s.id}
                onClick={() => setStrategy(s.id)}
                className="relative text-left rounded-2xl p-6 border transition-all duration-300 cursor-pointer hover:scale-[1.01]"
                style={{
                  background: selected ? `${s.color}12` : '#1a1a1a',
                  borderColor: selected ? s.color : 'rgba(255,255,255,0.07)',
                  boxShadow: selected ? `0 0 30px ${s.color}20` : 'none',
                  transform: selected ? 'scale(1.02)' : undefined,
                }}
              >
                {/* Selected indicator */}
                {selected && (
                  <div
                    className="absolute top-4 right-4 w-5 h-5 rounded-full flex items-center justify-center"
                    style={{ backgroundColor: s.color }}
                  >
                    <span className="text-noir-950 text-xs font-bold leading-none">+</span>
                  </div>
                )}

                {/* Icon badge */}
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center font-mono text-xs font-bold mb-4"
                  style={{
                    backgroundColor: `${s.color}20`,
                    color: s.color,
                    border: `1px solid ${s.color}30`,
                  }}
                >
                  {s.icon}
                </div>

                <p className="font-display text-xl text-cream mb-1">{s.label}</p>
                <p
                  className="font-mono text-xs uppercase tracking-widest mb-4"
                  style={{ color: s.color }}
                >
                  {s.subtitle}
                </p>
                <p className="font-body text-white/50 text-sm leading-relaxed mb-4">
                  {s.description}
                </p>
                <div
                  className="inline-block font-mono text-xs px-3 py-1 rounded-full border"
                  style={{ borderColor: `${s.color}40`, color: `${s.color}80` }}
                >
                  Best for: {s.bestFor}
                </div>

                {/* Alpha slider - fairness-aware only */}
                {s.hasSlider && selected && (
                  <div className="mt-6 pt-5 border-t border-white/10">
                    <div className="flex justify-between mb-2">
                      <span className="font-mono text-xs text-white/30">More fair</span>
                      <span className="font-mono text-xs text-amber-400">
                        alpha = {alpha.toFixed(1)}
                      </span>
                      <span className="font-mono text-xs text-white/30">More happy</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.1}
                      value={alpha}
                      onChange={(e) => setAlpha(parseFloat(e.target.value))}
                      onClick={(e) => e.stopPropagation()}
                      className="w-full accent-amber-400 cursor-pointer"
                    />
                    <p className="font-body text-white/30 text-xs mt-2 text-center">
                      {alpha < 0.4
                        ? 'Protecting the most dissatisfied member'
                        : alpha > 0.7
                        ? 'Maximizing total group satisfaction'
                        : 'Balanced - fair to everyone'}
                    </p>
                  </div>
                )}
              </button>
            )
          })}
        </div>

        <button
          onClick={() => navigate('/results')}
          className="group bg-amber-500 hover:bg-amber-400 text-noir-950 font-body font-semibold text-lg px-12 py-4 rounded-full transition-all duration-300 hover:scale-105"
          onMouseEnter={(e) => e.currentTarget.style.boxShadow = '0 0 40px rgba(245,158,11,0.4)'}
          onMouseLeave={(e) => e.currentTarget.style.boxShadow = 'none'}
        >
          Get Recommendations
          <span className="ml-3 inline-block transition-transform group-hover:translate-x-1">
            &rarr;
          </span>
        </button>
      </main>
    </div>
  )
}
