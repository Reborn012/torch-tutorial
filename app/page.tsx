import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Github, Linkedin, Mail, ExternalLink, ChevronDown } from "lucide-react"
import Link from "next/link"
import Image from "next/image"

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between py-4">
          <div className="flex gap-6 md:gap-10">
            <Link href="#" className="flex items-center space-x-2">
              <span className="font-bold inline-block">Portfolio</span>
            </Link>
          </div>
          <nav className="hidden gap-6 md:flex">
            <Link
              href="#about"
              className="flex items-center text-lg font-medium transition-colors hover:text-foreground/80 text-foreground/60"
            >
              About
            </Link>
            <Link
              href="#projects"
              className="flex items-center text-lg font-medium transition-colors hover:text-foreground/80 text-foreground/60"
            >
              Projects
            </Link>
            <Link
              href="#skills"
              className="flex items-center text-lg font-medium transition-colors hover:text-foreground/80 text-foreground/60"
            >
              Skills
            </Link>
            <Link
              href="#contact"
              className="flex items-center text-lg font-medium transition-colors hover:text-foreground/80 text-foreground/60"
            >
              Contact
            </Link>
          </nav>
          <Button variant="outline" className="bg-background text-foreground hidden md:flex">
            Download Resume
          </Button>
        </div>
      </header>
      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48">
          <div className="container px-4 md:px-6">
            <div className="grid gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_600px]">
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
                    Hi, I'm <span className="text-primary">Your Name</span>
                  </h1>
                  <p className="text-xl text-muted-foreground">
                    A passionate full-stack developer creating beautiful and functional web experiences
                  </p>
                </div>
                <div className="flex flex-col gap-2 min-[400px]:flex-row">
                  <Button>View My Work</Button>
                  <Button variant="outline" className="bg-background text-foreground">
                    Contact Me
                  </Button>
                </div>
                <div className="flex gap-4">
                  <Link
                    href="#"
                    className="rounded-full border border-muted p-2 text-muted-foreground hover:text-foreground"
                  >
                    <Github className="h-5 w-5" />
                    <span className="sr-only">GitHub</span>
                  </Link>
                  <Link
                    href="#"
                    className="rounded-full border border-muted p-2 text-muted-foreground hover:text-foreground"
                  >
                    <Linkedin className="h-5 w-5" />
                    <span className="sr-only">LinkedIn</span>
                  </Link>
                  <Link
                    href="#"
                    className="rounded-full border border-muted p-2 text-muted-foreground hover:text-foreground"
                  >
                    <Mail className="h-5 w-5" />
                    <span className="sr-only">Email</span>
                  </Link>
                </div>
              </div>
              <div className="flex items-center justify-center">
                <Image
                  src="/placeholder.svg?height=600&width=600"
                  alt="Hero Image"
                  width={600}
                  height={600}
                  className="rounded-lg object-cover"
                  priority
                />
              </div>
            </div>
            <div className="absolute bottom-6 left-1/2 hidden -translate-x-1/2 md:flex">
              <Link
                href="#about"
                className="flex items-center justify-center rounded-full p-2 text-muted-foreground hover:text-foreground"
              >
                <ChevronDown className="h-8 w-8 animate-bounce" />
                <span className="sr-only">Scroll Down</span>
              </Link>
            </div>
          </div>
        </section>
        <section id="about" className="w-full py-12 md:py-24 lg:py-32 bg-muted/40">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">About Me</h2>
                <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                  I'm a passionate developer with a strong background in web development. I love creating beautiful,
                  functional, and user-friendly websites and applications.
                </p>
              </div>
            </div>
            <div className="grid gap-6 lg:grid-cols-2 lg:gap-12 mt-8">
              <div className="flex justify-center">
                <Image
                  src="/placeholder.svg?height=400&width=400"
                  alt="About Me Image"
                  width={400}
                  height={400}
                  className="rounded-lg object-cover"
                />
              </div>
              <div className="flex flex-col justify-center space-y-4">
                <p className="text-base/relaxed md:text-lg/relaxed lg:text-base/relaxed xl:text-lg/relaxed">
                  With over X years of experience in web development, I've worked on a variety of projects ranging from
                  small business websites to complex web applications. I specialize in React, Next.js, and Node.js, and
                  I'm always eager to learn new technologies.
                </p>
                <p className="text-base/relaxed md:text-lg/relaxed lg:text-base/relaxed xl:text-lg/relaxed">
                  When I'm not coding, you can find me hiking, reading, or experimenting with new recipes in the
                  kitchen. I believe in continuous learning and pushing the boundaries of what's possible with web
                  technologies.
                </p>
                <div className="flex flex-col gap-2 min-[400px]:flex-row">
                  <Button variant="outline" className="bg-background text-foreground">
                    Download Resume
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section id="projects" className="w-full py-12 md:py-24 lg:py-32">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">My Projects</h2>
                <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                  Here are some of the projects I've worked on recently.
                </p>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 mt-8">
              <Card>
                <CardHeader>
                  <div className="overflow-hidden rounded-lg">
                    <Image
                      src="/placeholder.svg?height=300&width=500"
                      alt="Project 1"
                      width={500}
                      height={300}
                      className="object-cover transition-transform duration-300 hover:scale-105"
                    />
                  </div>
                  <CardTitle className="mt-4">Project One</CardTitle>
                  <CardDescription>
                    A responsive e-commerce website built with Next.js and Tailwind CSS.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    This project features a modern UI, shopping cart functionality, and integration with a headless CMS
                    for product management.
                  </p>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" className="bg-background text-foreground" size="sm">
                    <Github className="mr-2 h-4 w-4" />
                    Code
                  </Button>
                  <Button size="sm">
                    <ExternalLink className="mr-2 h-4 w-4" />
                    Demo
                  </Button>
                </CardFooter>
              </Card>
              <Card>
                <CardHeader>
                  <div className="overflow-hidden rounded-lg">
                    <Image
                      src="/placeholder.svg?height=300&width=500"
                      alt="Project 2"
                      width={500}
                      height={300}
                      className="object-cover transition-transform duration-300 hover:scale-105"
                    />
                  </div>
                  <CardTitle className="mt-4">Project Two</CardTitle>
                  <CardDescription>A task management application with real-time updates.</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Built with React, Firebase, and Tailwind CSS. Features include user authentication, task creation,
                    and real-time collaboration.
                  </p>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" className="bg-background text-foreground" size="sm">
                    <Github className="mr-2 h-4 w-4" />
                    Code
                  </Button>
                  <Button size="sm">
                    <ExternalLink className="mr-2 h-4 w-4" />
                    Demo
                  </Button>
                </CardFooter>
              </Card>
              <Card>
                <CardHeader>
                  <div className="overflow-hidden rounded-lg">
                    <Image
                      src="/placeholder.svg?height=300&width=500"
                      alt="Project 3"
                      width={500}
                      height={300}
                      className="object-cover transition-transform duration-300 hover:scale-105"
                    />
                  </div>
                  <CardTitle className="mt-4">Project Three</CardTitle>
                  <CardDescription>A blog platform with a custom CMS.</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Developed using Next.js, MongoDB, and Tailwind CSS. Features include markdown support, categories,
                    and a responsive design.
                  </p>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" className="bg-background text-foreground" size="sm">
                    <Github className="mr-2 h-4 w-4" />
                    Code
                  </Button>
                  <Button size="sm">
                    <ExternalLink className="mr-2 h-4 w-4" />
                    Demo
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </div>
        </section>
        <section id="skills" className="w-full py-12 md:py-24 lg:py-32 bg-muted/40">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">My Skills</h2>
                <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                  Here are some of the technologies and tools I work with.
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 mt-8">
              {[
                "HTML",
                "CSS",
                "JavaScript",
                "TypeScript",
                "React",
                "Next.js",
                "Node.js",
                "Express",
                "MongoDB",
                "PostgreSQL",
                "Tailwind CSS",
                "Git",
                "GitHub",
                "Figma",
                "Responsive Design",
              ].map((skill) => (
                <div
                  key={skill}
                  className="flex items-center justify-center rounded-lg border border-muted bg-background p-4 text-center"
                >
                  {skill}
                </div>
              ))}
            </div>
          </div>
        </section>
        <section id="contact" className="w-full py-12 md:py-24 lg:py-32">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">Get In Touch</h2>
                <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                  Have a project in mind or just want to say hello? Feel free to reach out!
                </p>
              </div>
            </div>
            <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 lg:grid-cols-2 lg:gap-12 mt-8">
              <div className="flex flex-col gap-4">
                <div className="flex items-start gap-4">
                  <Mail className="h-6 w-6 text-primary" />
                  <div className="grid gap-1">
                    <h3 className="text-xl font-bold">Email</h3>
                    <p className="text-muted-foreground">youremail@example.com</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <Linkedin className="h-6 w-6 text-primary" />
                  <div className="grid gap-1">
                    <h3 className="text-xl font-bold">LinkedIn</h3>
                    <p className="text-muted-foreground">linkedin.com/in/yourname</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <Github className="h-6 w-6 text-primary" />
                  <div className="grid gap-1">
                    <h3 className="text-xl font-bold">GitHub</h3>
                    <p className="text-muted-foreground">github.com/yourusername</p>
                  </div>
                </div>
              </div>
              <div className="flex flex-col gap-4">
                <div className="grid gap-2">
                  <label htmlFor="name" className="font-medium">
                    Name
                  </label>
                  <input
                    id="name"
                    className="rounded-md border border-input bg-background px-3 py-2 text-sm"
                    placeholder="Your name"
                  />
                </div>
                <div className="grid gap-2">
                  <label htmlFor="email" className="font-medium">
                    Email
                  </label>
                  <input
                    id="email"
                    type="email"
                    className="rounded-md border border-input bg-background px-3 py-2 text-sm"
                    placeholder="Your email"
                  />
                </div>
                <div className="grid gap-2">
                  <label htmlFor="message" className="font-medium">
                    Message
                  </label>
                  <textarea
                    id="message"
                    className="rounded-md border border-input bg-background px-3 py-2 text-sm min-h-[120px]"
                    placeholder="Your message"
                  />
                </div>
                <Button>Send Message</Button>
              </div>
            </div>
          </div>
        </section>
      </main>
      <footer className="w-full border-t py-6 md:py-0">
        <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
          <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
            © 2024 Your Name. All rights reserved.
          </p>
          <div className="flex gap-4">
            <Link href="#" className="rounded-full border border-muted p-2 text-muted-foreground hover:text-foreground">
              <Github className="h-5 w-5" />
              <span className="sr-only">GitHub</span>
            </Link>
            <Link href="#" className="rounded-full border border-muted p-2 text-muted-foreground hover:text-foreground">
              <Linkedin className="h-5 w-5" />
              <span className="sr-only">LinkedIn</span>
            </Link>
            <Link href="#" className="rounded-full border border-muted p-2 text-muted-foreground hover:text-foreground">
              <Mail className="h-5 w-5" />
              <span className="sr-only">Email</span>
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
