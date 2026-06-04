import React from 'react'
import Modal from './Modal'
import type { PersonaSnapshot } from '../../types'

interface Props {
  persona: PersonaSnapshot
  onClose: () => void
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <p className="font-mono text-[13px] font-bold tracking-[0.15em] uppercase text-faint mb-3">{children}</p>
}

function MetaCell({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <p className="font-mono text-[11px] tracking-widest uppercase text-faint mb-0.5">{label}</p>
      <p className="text-[15px] text-ink">{value}</p>
    </div>
  )
}

export default function PersonaModal({ persona, onClose }: Props) {
  const notes = persona.notes ?? {}
  const noteFields: { key: string; val: string }[] = []

  if (notes.tone)     noteFields.push({ key: 'tone', val: notes.tone })
  if (notes.language) noteFields.push({ key: 'language', val: notes.language })
  if (notes.should_contradict) noteFields.push({ key: 'contradictory', val: 'yes' })
  if (notes.extra)    noteFields.push({ key: 'extra', val: notes.extra })

  const knownKeys = new Set(['tone', 'language', 'should_contradict', 'extra'])
  for (const [k, v] of Object.entries(notes)) {
    if (!knownKeys.has(k) && v !== null && v !== undefined && v !== false && v !== '') {
      noteFields.push({ key: k, val: String(v) })
    }
  }

  return (
    <Modal title="Persona" onClose={onClose} width="max-w-lg">
      <div className="px-8 py-7 flex flex-col gap-6">

        {/* Name + description */}
        <div>
          <h3 className="font-serif text-[28px] text-ink leading-tight">{persona.name}</h3>
          {persona.description && (
            <p className="text-[15px] text-faint leading-relaxed mt-1.5">{persona.description}</p>
          )}
        </div>

        <div className="h-px bg-border" />

        {/* Goal */}
        <div>
          <SectionLabel>Goal</SectionLabel>
          <p className="text-[16px] text-muted leading-relaxed">{persona.goal}</p>
        </div>

        <div className="h-px bg-border" />

        {/* Meta strip */}
        <div className="flex items-center gap-6">
          <MetaCell label="Dataset" value={persona.dataset} />
          <div className="w-px h-8 bg-border" />
          <MetaCell label="Initial k" value={<span className="font-mono">{persona.k_initial}</span>} />
          {persona.model && (
            <>
              <div className="w-px h-8 bg-border" />
              <MetaCell label="Model" value={persona.model} />
            </>
          )}
        </div>

        {/* Notes */}
        {noteFields.length > 0 && (
          <>
            <div className="h-px bg-border" />
            <div>
              <SectionLabel>Notes</SectionLabel>
              <div className="flex flex-col gap-4">
                {noteFields.map(({ key, val }) => (
                  <div key={key}>
                    <span className="font-mono text-[11px] tracking-widest uppercase text-faint">{key}</span>
                    <p className="text-[15px] text-muted leading-relaxed mt-0.5">{val}</p>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

      </div>
    </Modal>
  )
}
