FROM golang:1.20

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

ENV INFERENCE_HOSTNAME=inference:5000
RUN CGO_ENABLED=0 GOOS=linux go build -o server ./bin/server 

EXPOSE 3000

CMD ["./server"]
