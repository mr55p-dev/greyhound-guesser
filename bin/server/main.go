package main

import (
	"bytes"
	"log"
	"net/http"
	"text/template"

	"github.com/labstack/echo"
)

type FormControls struct {
	TrapNumbers []int
}

func ReadTemplate(data any, files ...string) string {
	out := new(bytes.Buffer)
	t := template.Must(
		template.ParseFiles(files...),
	)
	t.Execute(out, data)
	return out.String()
}

func main() {
	e := echo.New()
	e.Logger.Info("Starting server")
	e.Static("/", "public")
	e.GET("/", func(c echo.Context) error {
		c.Logger().Debug("Index request received")
		html := ReadTemplate(
			FormControls{TrapNumbers: []int{0, 1, 2, 3, 4, 5}},
			"templates/index.html",
			"templates/common.html",
		)
		return c.HTML(http.StatusOK, html)
	})

	e.POST("/submit", func(c echo.Context) error {
		c.Logger().Debug("Submit request received")
		html := ReadTemplate(
			nil,
			"templates/results.html",
			"templates/common.html",
		)
		return c.HTML(http.StatusOK, html)
	})

	log.Fatal(e.Start(":3000"))
}

curl "http://localhost:5000/predict" --data-raw 'dog-0-odds=1&dog-0-finished=1&dog-0-distance=1&dog-1-odds=1&dog-1-finished=1&dog-1-distance=1&dog-2-odds=1&dog-2-finished=1&dog-2-distance=1&dog-3-odds=1&dog-3-finished=1&dog-3-distance=1&dog-4-odds=1&dog-4-finished=1&dog-4-distance=1&dog-5-odds=1&dog-5-finished=1&dog-5-distance=1'
