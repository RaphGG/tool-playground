const net = require("net");
/*
let socket = new net.Socket();

socket = socket.connect(6000, "192.168.1.11", () => {
  console.log("Connected to Switch!");
});

socket.write('click HOME\r\n', 'utf8', (err) => {
  console.log(err);
});
*/

const socket = net.connect(6000, "192.168.1.11", () => {
  console.log("Connected to Switch!");
  socket.write('configure echoCommands 1\r\n', 'utf8');
});

socket.on("data", (data) => {
  console.log(data);
  console.log(socket.remoteAddress);
  //socket.destroy();
});

socket.on('close', () => {
  console.log('Disconnected from switch!');
});

setTimeout(() => {
  socket.write('click A\r\n');
}, 10000)


/*
const connect = async () => {
  return await net.connect(6000, '192.168.1.11', () => {
    console.log("connected?");
  });
};
*/
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