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
			FormControls{TrapNumbers: []int{1, 2, 3, 4, 5, 6}},
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
