import BrawlerIcon from "./brawler-icon";

export default function BrawlerPicker() {
    {/* hier dann brawler importieren und später ai-logik mit integrieren*/}
    const brawlers:string[] = []
    return (
      <div className="grid grid-cols-3 gap-2 p-4">
        {brawlers.map((brawler) => (
          <BrawlerIcon key={brawler.id} brawler={brawler} />
        ))}
      </div>
    );
  }