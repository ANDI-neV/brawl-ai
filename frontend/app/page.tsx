import Image from "next/image";
import Header from "./components/header"
import TeamDisplay from "./components/team-display";

export default function Home() {
  return (
    <main className="">
      <Header/>
      <div className="flex h-screen">
        <div className="w-1/6">
          <TeamDisplay team="left" bgColor="#3b82f6"/>
        </div>
        <div className="w-2/6">
          <TeamDisplay team="right" bgColor="#f43f5e"/>
        </div>
      </div>
    </main>
  );
}
