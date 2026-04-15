import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts'

const API = import.meta.env.VITE_API_URL || '/api'
const TMDB_IMAGE = 'https://image.tmdb.org/t/p/w300'
const STEP_LABELS = ['Group', 'Rate', 'Strategy', 'Results']

const AVATAR_COLORS = [
  '#f59e0b', '#10b981', '#3b82f6', '#ec4899',
  '#8b5cf6', '#f97316', '#06b6d4', '#84cc16',
  '#ef4444', '#a855f7',
]

const STRATEGY_LABELS = {
  least_misery: 'No One Hates It',
  average: 'Most Overall Happy',
  fairness_aware: 'The Balanced Pick',
}

/**
 * MemberBar
 * Horizontal satisfaction bar for one member on one recommended movie.
 */
function MemberBar({ name, score, color, delayIndex }) {
  const pct = ((score - 0.5) / 4.5) * 100

  return (
    <div className="flex items-center gap-2 mb-2">
      <div
        className="w-5 h-5 rounded-full flex items-center justify-center font-bold flex-shrink-0"
        style={{ fontSize: '9px', backgroundColor: color, color: '#080808' }}
      >
        {name[0]?.toUpperCase()}
      </div>
      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ backgroundColor: '#242424' }}>
        <div
          className="h-2 rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            backgroundColor: color,
            transitionDelay: `${delayIndex * 80}ms`,
          }}
        />
      </div>
      <span className="font-mono text-xs text-white/30 w-6 text-right">
        {score.toFixed(1)}
      </span>
    </div>
  )
}

/**
 * Results page.
 * Calls /recommend, renders recommendation cards with expandable
 * per-member satisfaction bars, plus a group fairness dashboard.
 */
export default function Results({ results, setResults, members, strategy, alpha }) {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(null)

  useEffect(() => {
    if (!members?.length) {
      navigate('/')
      return
    }
    fetchRecommendations()
  }, [])

  const fetchRecommendations = async () => {
    setLoading(true)
    setError(null)
    try {
      const payload = {
        members: members.map((m) => ({
          member_id: m.id,
          ratings: m.ratings,
        })),
        strategy,
        alpha,
        top_k: 10,
        model: 'ncf',
      }
      const res = await axios.post(`${API}/recommend`, payload)
      setResults(res.data)
    } catch (err) {
      console.error(err)
      setError(
        'Could not reach the recommendation server. Make sure the API is running on port 8000.'
      )
    } finally {
      setLoading(false)
    }
  }

  // Build radar chart data from per-member satisfaction scores
  const radarData = results
    ? members.map((m) => ({
        name: m.name,
        satisfaction: Math.round(
          (results.fairness_summary.per_member_satisfaction[m.id] || 0) * 20
        ),
      }))
    : []

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
              <span className={`font-mono text-xs uppercase tracking-widest ${i === 3 ? 'text-amber-400' : 'text-white/20'}`}>
                {step}
              </span>
              {i < STEP_LABELS.length - 1 && (
                <span className="text-white/10 text-xs">&rsaquo;</span>
              )}
            </React.Fragment>
          ))}
        </div>
      </nav>

      <main className="flex-1 px-6 py-12 max-w-6xl mx-auto w-full">

        {/* Loading state */}
        {loading && (
          <div className="flex flex-col items-center justify-center h-96 gap-6">
            <div className="relative w-16 h-16">
              <div
                className="absolute inset-0 rounded-full border-2"
                style={{ borderColor: 'rgba(245,158,11,0.2)' }}
              />
              <div
                className="absolute inset-0 rounded-full border-2 border-t-amber-400 animate-spin"
                style={{ borderColor: 'transparent', borderTopColor: '#f59e0b' }}
              />
            </div>
            <div className="text-center">
              <p className="font-display text-2xl text-cream mb-2">
                Finding your film...
              </p>
              <p className="font-mono text-xs text-white/30 uppercase tracking-widest">
                Aggregating {members.length} taste profiles
              </p>
            </div>
          </div>
        )}

        {/* Error state */}
        {error && !loading && (
          <div className="flex flex-col items-center justify-center h-64 gap-4">
            <p className="font-body text-red-400 text-center max-w-md">{error}</p>
            <button
              onClick={fetchRecommendations}
              className="bg-amber-500 text-noir-950 font-semibold px-8 py-3 rounded-full hover:bg-amber-400 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Results */}
        {!loading && !error && results && (
          <>
            {/* Page header */}
            <div className="mb-10 fade-up">
              <p className="font-mono text-amber-400 text-xs uppercase tracking-widest mb-2">
                Step 4 of 4 &middot; Strategy: {STRATEGY_LABELS[strategy]}
              </p>
              <h2 className="font-display text-4xl md:text-5xl text-cream leading-tight">
                Tonight's picks for{' '}
                {members.map((m) => m.name).join(', ')}
              </h2>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

              {/* Left column: recommendation cards */}
              <div className="lg:col-span-2 space-y-3">
                {results.recommendations.map((rec, recIdx) => {
                  const isExpanded = expanded === recIdx
                  const isTopPick = recIdx === 0

                  return (
                    <div
                      key={rec.movie.movie_id}
                      className="rounded-2xl overflow-hidden border transition-all duration-300 cursor-pointer"
                      style={{
                        backgroundColor: '#1a1a1a',
                        borderColor: isTopPick
                          ? 'rgba(245,158,11,0.4)'
                          : 'rgba(255,255,255,0.05)',
                        boxShadow: isTopPick
                          ? '0 0 20px rgba(245,158,11,0.1)'
                          : 'none',
                      }}
                      onClick={() => setExpanded(isExpanded ? null : recIdx)}
                    >
                      <div className="flex gap-4 p-4">

                        {/* Rank number */}
                        <div className="flex-shrink-0 w-10 flex flex-col items-center justify-center">
                          {isTopPick ? (
                            <span
                              className="font-display text-lg"
                              style={{ color: '#f59e0b' }}
                            >
                              #1
                            </span>
                          ) : (
                            <span className="font-mono text-white/20 text-sm">
                              #{recIdx + 1}
                            </span>
                          )}
                        </div>

                        {/* Movie poster */}
                        <div
                          className="flex-shrink-0 w-14 h-20 rounded-lg overflow-hidden"
                          style={{ backgroundColor: '#242424' }}
                        >
                          {rec.movie.tmdb_id ? (
                            <img
                              src={`${TMDB_IMAGE}${rec.movie.tmdb_id}`}
                              alt={rec.movie.title}
                              className="w-full h-full object-cover"
                              onError={(e) => { e.target.style.display = 'none' }}
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              <span className="font-mono text-white/10 text-xs">no img</span>
                            </div>
                          )}
                        </div>

                        {/* Title and genre */}
                        <div className="flex-1 min-w-0">
                          <h3 className="font-display text-lg text-cream leading-tight mb-1">
                            {rec.movie.title}
                          </h3>
                          <p className="font-mono text-xs text-white/30 mb-3">
                            {rec.movie.genres.join(' / ')}
                          </p>

                          {/* Quick member satisfaction preview */}
                          <div className="flex gap-1">
                            {rec.member_scores.map((ms) => {
                              const memberIdx = members.findIndex(
                                (m) => m.id === ms.member_id
                              )
                              return (
                                <div
                                  key={ms.member_id}
                                  title={`${members[memberIdx]?.name}: ${ms.predicted_rating.toFixed(1)}/5`}
                                  className="h-1.5 rounded-full"
                                  style={{
                                    width: '20px',
                                    backgroundColor: AVATAR_COLORS[memberIdx],
                                    opacity: 0.3 + (ms.predicted_rating / 5) * 0.7,
                                  }}
                                />
                              )
                            })}
                          </div>
                        </div>

                        {/* Group score */}
                        <div className="flex-shrink-0 flex flex-col items-center justify-center px-2">
                          <span
                            className="font-display text-2xl"
                            style={{ color: '#f59e0b' }}
                          >
                            {rec.group_score.toFixed(1)}
                          </span>
                          <span className="font-mono text-xs text-white/20">group</span>
                        </div>

                        {/* Expand toggle */}
                        <div className="flex-shrink-0 flex items-center pl-1">
                          <span
                            className="text-white/20 text-sm transition-transform duration-200 inline-block"
                            style={{
                              transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                            }}
                          >
                            v
                          </span>
                        </div>
                      </div>

                      {/* Expanded per-member breakdown */}
                      {isExpanded && (
                        <div
                          className="px-6 pb-5 pt-4"
                          style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
                        >
                          <p className="font-mono text-xs uppercase tracking-widest text-white/30 mb-3">
                            Per-member satisfaction
                          </p>
                          {rec.member_scores.map((ms, mi) => {
                            const memberObj = members.find((m) => m.id === ms.member_id)
                            const memberIdx = members.findIndex((m) => m.id === ms.member_id)
                            return (
                              <MemberBar
                                key={ms.member_id}
                                name={memberObj?.name || ms.member_id}
                                score={ms.predicted_rating}
                                color={AVATAR_COLORS[memberIdx]}
                                delayIndex={mi}
                              />
                            )
                          })}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>

              {/* Right column: fairness dashboard */}
              <div className="space-y-4">

                {/* Summary metrics */}
                <div
                  className="rounded-2xl p-6 border"
                  style={{ backgroundColor: '#1a1a1a', borderColor: 'rgba(255,255,255,0.05)' }}
                >
                  <p className="font-mono text-xs uppercase tracking-widest text-white/30 mb-5">
                    Group Fairness
                  </p>

                  <div className="flex justify-around mb-6">
                    <div className="text-center">
                      <p
                        className="font-display text-3xl"
                        style={{ color: '#f59e0b' }}
                      >
                        {results.fairness_summary.avg_group_satisfaction.toFixed(1)}
                      </p>
                      <p className="font-mono text-xs text-white/30 mt-1">
                        avg satisfaction
                      </p>
                    </div>
                    <div className="text-center">
                      <p
                        className="font-display text-3xl"
                        style={{ color: '#10b981' }}
                      >
                        {Math.round(results.fairness_summary.fairness_score * 100)}%
                      </p>
                      <p className="font-mono text-xs text-white/30 mt-1">
                        fairness score
                      </p>
                    </div>
                  </div>

                  {/* Radar chart - only render with 3+ members */}
                  {radarData.length >= 3 && (
                    <ResponsiveContainer width="100%" height={200}>
                      <RadarChart data={radarData}>
                        <PolarGrid stroke="#2e2e2e" />
                        <PolarAngleAxis
                          dataKey="name"
                          tick={{
                            fill: 'rgba(255,255,255,0.3)',
                            fontSize: 11,
                            fontFamily: 'DM Mono',
                          }}
                        />
                        <Radar
                          dataKey="satisfaction"
                          stroke="#f59e0b"
                          fill="#f59e0b"
                          fillOpacity={0.15}
                          strokeWidth={2}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  )}

                  {/* Per-member satisfaction bars */}
                  <div className="space-y-2 mt-4">
                    {members.map((m, i) => {
                      const sat =
                        results.fairness_summary.per_member_satisfaction[m.id] || 0
                      const pct = ((sat - 0.5) / 4.5) * 100
                      return (
                        <div key={m.id} className="flex items-center gap-3">
                          <div
                            className="w-6 h-6 rounded-full flex items-center justify-center font-bold flex-shrink-0"
                            style={{
                              fontSize: '10px',
                              backgroundColor: AVATAR_COLORS[i],
                              color: '#080808',
                            }}
                          >
                            {m.name[0]?.toUpperCase()}
                          </div>
                          <div
                            className="flex-1 h-1.5 rounded-full overflow-hidden"
                            style={{ backgroundColor: '#242424' }}
                          >
                            <div
                              className="h-1.5 rounded-full transition-all duration-700"
                              style={{
                                width: `${pct}%`,
                                backgroundColor: AVATAR_COLORS[i],
                              }}
                            />
                          </div>
                          <span className="font-mono text-xs text-white/30">
                            {sat.toFixed(1)}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Navigation options */}
                <div
                  className="rounded-2xl p-5 border space-y-3"
                  style={{ backgroundColor: '#1a1a1a', borderColor: 'rgba(255,255,255,0.05)' }}
                >
                  <p className="font-mono text-xs uppercase tracking-widest text-white/30">
                    Want different results?
                  </p>
                  <button
                    onClick={() => navigate('/strategy')}
                    className="w-full font-body font-medium py-3 rounded-xl transition-colors text-sm"
                    style={{
                      border: '1px solid rgba(245,158,11,0.3)',
                      color: '#f59e0b',
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(245,158,11,0.08)'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                  >
                    Change Strategy
                  </button>
                  <button
                    onClick={() => navigate('/')}
                    className="w-full font-body font-medium py-3 rounded-xl transition-colors text-sm"
                    style={{
                      border: '1px solid rgba(255,255,255,0.08)',
                      color: 'rgba(255,255,255,0.4)',
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(255,255,255,0.7)'}
                    onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(255,255,255,0.4)'}
                  >
                    Start Over
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
