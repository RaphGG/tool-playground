const net = require("net");




const connect = async () => {
  return await net.connect(6000, '192.168.1.11', () => {
    console.log("connected?");
  });
};

/*
const socket = net.connect(6000, "192.168.1.11", () => {
  console.log("Connected to Switch!");
});


socket.on("data", (data) => {
  console.log(data.toString());
  console.log(socket.remoteAddress);
  //socket.end();
});

socket.on('end', () => {
  console.log('Disconnected from switch!');
});
*/