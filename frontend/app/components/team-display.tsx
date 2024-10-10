import React from 'react';
import BrawlerSlot from "./brawler-slot";

interface TeamDisplayProps {
  team: 'left' | 'right';
  bgColor: string;
}

export default function TeamDisplay({ team, bgColor }: TeamDisplayProps) {
  const indices = team === 'left' ? [0, 1, 2] : [3, 4, 5];
  return (
    <div className="flex flex-row items-center justify-center gap-3">
      {indices.map((index) => (
        <BrawlerSlot key={index} index={index} team={team} bgColor={bgColor}/>
      ))}
    </div>
  );
}