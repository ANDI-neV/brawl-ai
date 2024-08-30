import BrawlerSlot from "./brawler-slot";

interface TeamDisplayProps {
  team: 'left' | 'right';
  bgColor: string;
}

export default function TeamDisplay({ team, bgColor }: TeamDisplayProps) {
  const indices = team === 'left' ? [0, 1, 2] : [3, 4, 5];
  return (
    <div className="flex flex-col items-center p-4 gap-3">
      {indices.map((index) => (
        <BrawlerSlot index={index} bgColor={bgColor}/>
      ))}
    </div>
  );
}