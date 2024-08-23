import Image from "next/image";

interface BrawlerSlotProps {
    team: 'left' | 'right';
    index: number;
    bgColor: string;
  }
  
  export default function BrawlerSlot({ team, index, bgColor }: BrawlerSlotProps) {
    return (
      <div className="h-1/3 border p-3 rounded-xl w-[125px]"
      style={{backgroundColor: bgColor}}
      >
        { 
            <Image src="/brawlstar.png"
                alt="brawlstar"
                width={100}
                height={100}
            />
        }
      </div>
    );
  }