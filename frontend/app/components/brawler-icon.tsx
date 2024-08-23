interface BrawlerIconProps {
    brawler: Brawler;
  }
  
  export default function BrawlerIcon({ brawler }: BrawlerIconProps) {
    return (
      <button className="w-full aspect-square" onClick={() => selectBrawler(brawler)}>
        <img src={brawler.imageUrl} alt={brawler.name} className="w-full h-full object-cover" />
      </button>
    );
  }

  function selectBrawler({brawler}: BrawlerIconProps) {}