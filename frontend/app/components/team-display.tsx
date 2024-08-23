import BrawlerSlot from "./brawler-slot";

interface TeamDisplayProps {
    team: 'left' | 'right';
    bgColor: string;
  }
  
  export default function TeamDisplay({ team, bgColor }: TeamDisplayProps) {
    return (
      <div className="flex flex-col items-center">
        {[0, 1, 2].map((index) => (
          <BrawlerSlot key={index} team={team} index={index} bgColor={bgColor}/>
        ))}
      </div>
    );
  }